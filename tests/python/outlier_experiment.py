#!/usr/bin/env python3
"""Outlier detection comparison: EndTree vs LOF on multiple datasets.

This script:
- Tests on MNIST, GloVe, SIFT, and Random datasets
- Builds EndTree on clean training data (inliers only)
- Injects synthetic outliers into test set
- Compares outlier detection accuracy between EndTree and LOF
- Reports precision, recall, and true positive rates

Run:
  python3 -m tests.python.outlier_experiment --dataset mnist
  python3 -m tests.python.outlier_experiment --dataset glove --glove-dim 100
  python3 -m tests.python.outlier_experiment --dataset sift
  python3 -m tests.python.outlier_experiment --dataset random --dim 128
  python3 -m tests.python.outlier_experiment --dataset all
"""

from __future__ import annotations

import gzip
import argparse
import struct
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import zipfile

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import urllib.request
from typing import  IO, List, Optional, Sequence, Tuple


# Add parent directory to path to import bthenn module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import hnswlib

MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = {
	"train_images": "train-images-idx3-ubyte.gz",
	"train_labels": "train-labels-idx1-ubyte.gz",
	"test_images": "t10k-images-idx3-ubyte.gz",
	"test_labels": "t10k-labels-idx1-ubyte.gz",
}

GLOVE_ZIP_URL = "https://nlp.stanford.edu/data/glove.6B.zip"


def _cache_dir(name: str) -> Path:
	path = Path.home() / ".cache" / "bthenn" / name
	path.mkdir(parents=True, exist_ok=True)
	return path


def _download(url: str, dest: Path) -> None:
	dest.parent.mkdir(parents=True, exist_ok=True)
	with urllib.request.urlopen(url) as response, open(dest, "wb") as f:
		f.write(response.read())


def _download_any(urls: Sequence[str], dest: Path) -> None:
	last_err: Optional[BaseException] = None
	for url in urls:
		try:
			_download(url, dest)
			return
		except BaseException as e:
			last_err = e
	raise RuntimeError(f"Failed to download {dest.name} from any known URL") from last_err

# reference: https://www.kaggle.com/code/hojjatk/read-mnist-dataset
def _read_idx_images_gz(path: Path) -> np.ndarray:
	with gzip.open(path, "rb") as f:
		header = f.read(16)
		magic, num, rows, cols = struct.unpack(">IIII", header)
		if magic != 2051:
			raise ValueError(f"Bad MNIST image magic {magic} in {path}")
		data = f.read(rows * cols * num)
	images = np.frombuffer(data, dtype=np.uint8)
	return images.reshape(num, rows * cols)


def _read_idx_labels_gz(path: Path) -> np.ndarray:
	with gzip.open(path, "rb") as f:
		header = f.read(8)
		magic, num = struct.unpack(">II", header)
		if magic != 2049:
			raise ValueError(f"Bad MNIST label magic {magic} in {path}")
		data = f.read(num)
	labels = np.frombuffer(data, dtype=np.uint8)
	return labels


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	cache_dir = _cache_dir("mnist")
	paths = {k: cache_dir / v for k, v in MNIST_FILES.items()}
	for v in MNIST_FILES.values():
		url = MNIST_BASE_URL + v
		dest = cache_dir / v
		dest.parent.mkdir(parents=True, exist_ok=True)
		if dest.exists() and dest.stat().st_size > 0:
			continue
		_download(url, dest)

	x_train_u8 = _read_idx_images_gz(paths["train_images"])
	y_train_u8 = _read_idx_labels_gz(paths["train_labels"])
	x_test_u8 = _read_idx_images_gz(paths["test_images"])
	y_test_u8 = _read_idx_labels_gz(paths["test_labels"])
	return x_train_u8, y_train_u8, x_test_u8, y_test_u8


def _read_fvecs_partial(path: Path, n: int) -> np.ndarray:
	"""Read first n vectors from a .fvecs file into float32 array [n, d]."""
	if n <= 0:
		return np.zeros((0, 0), dtype=np.float32)
	with open(path, "rb") as f:
		d = np.fromfile(f, dtype=np.int32, count=1)
		if d.size != 1:
			raise ValueError(f"Empty/invalid fvecs file: {path}")
		dim = int(d[0])
		f.seek(0)
		count = int(n) * int(dim + 1)
		raw = np.fromfile(f, dtype=np.float32, count=count)
	if raw.size % (dim + 1) != 0:
		# File ended early; still return what we have.
		raw = raw[: (raw.size // (dim + 1)) * (dim + 1)]
	arr = raw.reshape(-1, dim + 1)[:, 1:]
	return arr.astype(np.float32, copy=False)


def _load_glove_vectors_from_text(f: IO[bytes], *, max_rows: int, dim: int) -> np.ndarray:
	vectors: List[np.ndarray] = []
	for i, raw_line in enumerate(f):
		if i >= max_rows:
			break
		line = raw_line.decode("utf-8", errors="ignore") if isinstance(raw_line, (bytes, bytearray)) else str(raw_line)
		parts = line.strip().split()
		if len(parts) != dim + 1:
			continue
		vec = np.asarray(parts[1:], dtype=np.float32)
		if vec.shape[0] != dim:
			continue
		vectors.append(vec)
	if not vectors:
		raise RuntimeError("No valid GloVe vectors parsed")
	return np.vstack(vectors).astype(np.float32, copy=False)


def load_glove(*, n_train: int, n_query: int, dim: int) -> Tuple[np.ndarray, np.ndarray]:
	"""Load a small prefix of glove.6B.{dim}d vectors for train/query."""
	max_rows = int(n_train) + int(n_query)
	if max_rows <= 0:
		raise ValueError("n_train + n_query must be > 0")
	if dim not in (50, 100, 200, 300):
		raise ValueError("glove_dim must be one of 50/100/200/300")

	# Prefer an explicit path if provided, otherwise try ./datasets then cache.
	cache_dir = _cache_dir("glove")
	zip_path = cache_dir / "glove.6B.zip"
	if (not zip_path.exists() or zip_path.stat().st_size == 0):
		_download(GLOVE_ZIP_URL, zip_path)
	if not zip_path.exists():
		raise FileNotFoundError(
			"Missing GloVe data. Provide --glove-path or re-run with --download-datasets "
			"to fetch glove.6B.zip into ~/.cache/bthenn/glove/."
		)
	inner_name = f"glove.6B.{int(dim)}d.txt"
	with zipfile.ZipFile(zip_path, "r") as zf:
		if inner_name not in zf.namelist():
			raise FileNotFoundError(f"{inner_name} not found inside {zip_path}")
		with zf.open(inner_name, "r") as f:
			all_vecs = _load_glove_vectors_from_text(f, max_rows=max_rows, dim=int(dim))

	if all_vecs.shape[0] < max_rows:
		raise RuntimeError(f"Not enough GloVe vectors loaded: needed {max_rows}, got {all_vecs.shape[0]}")
	x_train = all_vecs[: int(n_train)]
	x_query = all_vecs[int(n_train) : int(n_train) + int(n_query)]
	return x_train, x_query


def load_sift(*, n_train: int, n_query: int) -> Tuple[np.ndarray, np.ndarray]:
	"""Load SIFT1M base/query vectors (partial read)."""
	# Prefer ./datasets/sift, else ~/.cache/bthenn/sift, else an explicit --sift-dir.
	base_dir = Path("datasets") / "sift"
	if not base_dir.exists():
		base_dir = _cache_dir("sift")

	base_path = base_dir / "sift_base.fvecs"
	query_path = base_dir / "sift_query.fvecs"
	if (not base_path.exists() or not query_path.exists()):
		base_dir.mkdir(parents=True, exist_ok=True)
		# Download and extract sift.tar.gz
		tar_path = base_dir / "sift.tar.gz"
		if not tar_path.exists():
			# Try multiple mirrors for sift.tar.gz
			candidates = [
				"ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
				"http://corpus-texmex.irisa.fr/texmex/corpus/sift.tar.gz",
			]
			_download_any(candidates, tar_path)
		# Extract the tar.gz file
		if tar_path.exists() and (not base_path.exists() or not query_path.exists()):
			with tarfile.open(tar_path, "r:gz") as tar:
				tar.extractall(path=base_dir)
		
		# After extraction, the files should be in base_dir/sift/
		extracted_dir = base_dir / "sift"
		if extracted_dir.exists():
			if not base_path.exists() and (extracted_dir / "sift_base.fvecs").exists():
				base_path = extracted_dir / "sift_base.fvecs"
			if not query_path.exists() and (extracted_dir / "sift_query.fvecs").exists():
				query_path = extracted_dir / "sift_query.fvecs"

	if not base_path.exists() or not query_path.exists():
		raise FileNotFoundError(
			f"Missing SIFT data. Download sift.tar.gz and extract to {base_dir}/ "
			f"or put sift_base.fvecs and sift_query.fvecs directly under {base_dir}/"
		)

	x_train = _read_fvecs_partial(base_path, int(n_train))
	x_query = _read_fvecs_partial(query_path, int(n_query))
	if x_train.size == 0 or x_query.size == 0:
		raise RuntimeError("Failed to load SIFT vectors (empty arrays)")
	return x_train, x_query

def inject_outliers(
    x_train: np.ndarray,
    x_test: np.ndarray,
    *,
    outlier_ratio: float,
    outlier_type: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inject outliers into test set and return combined test data + labels.
    
    Args:
        x_train: Training data (clean, used for statistics)
        x_test: Original test data (inliers)
        outlier_ratio: Fraction of test set to replace with outliers
        outlier_type: 'uniform' or 'gaussian' outliers
        rng: Random number generator
    
    Returns:
        (test_data, labels) where labels[i] = 1 if outlier, 0 if inlier
    """
    n_test = x_test.shape[0]
    dim = x_test.shape[1]
    n_outliers = int(n_test * outlier_ratio)
    n_inliers = n_test - n_outliers
    
    # Sample inliers from original test set
    inlier_idx = rng.choice(n_test, size=n_inliers, replace=False)
    inliers = x_test[inlier_idx]
    
    # Generate outliers based on type
    if outlier_type == "uniform":
        # Uniform outliers in extended range
        data_min = np.min(x_train, axis=0)
        data_max = np.max(x_train, axis=0)
        data_range = data_max - data_min
        # Expand range by 50% on each side
        expanded_min = data_min - 0.5 * data_range
        expanded_max = data_max + 0.5 * data_range
        outliers = rng.uniform(
            low=expanded_min,
            high=expanded_max,
            size=(n_outliers, dim)
        ).astype(np.float32)
    elif outlier_type == "gaussian":
        # Gaussian outliers with high variance
        data_mean = np.mean(x_train, axis=0)
        data_std = np.std(x_train, axis=0)
        # Use 3x standard deviation for outliers
        outliers = rng.normal(
            loc=data_mean,
            scale=3.0 * data_std,
            size=(n_outliers, dim)
        ).astype(np.float32)
    else:
        raise ValueError(f"Unknown outlier_type: {outlier_type}")
    
    # Combine and shuffle
    combined_data = np.vstack([inliers, outliers]).astype(np.float32)
    labels = np.hstack([
        np.zeros(n_inliers, dtype=np.int32),  # 0 = inlier
        np.ones(n_outliers, dtype=np.int32),   # 1 = outlier
    ])
    
    # Shuffle
    shuffle_idx = rng.permutation(len(labels))
    combined_data = combined_data[shuffle_idx]
    labels = labels[shuffle_idx]
    
    return combined_data, labels


def calculate_metrics(
    scores: np.ndarray,
    true_labels: np.ndarray,
    top_k: int,
) -> Tuple[float, float, float]:
    """Calculate precision, recall, and true positive rate for top-k outliers.
    
    Args:
        scores: Outlier scores (higher = more outlier)
        true_labels: Ground truth (1 = outlier, 0 = inlier)
        top_k: Number of top outliers to consider
    
    Returns:
        (precision, recall, true_positive_rate)
    """
    n_total = len(true_labels)
    n_true_outliers = int(np.sum(true_labels))
    
    if n_true_outliers == 0:
        return 0.0, 0.0, 0.0
    
    # Get top-k indices by score
    top_k_eff = min(top_k, n_total)
    top_indices = np.argsort(scores)[-top_k_eff:][::-1]
    
    # Calculate metrics
    predicted_outliers = top_indices
    true_outliers = np.where(true_labels == 1)[0]
    
    true_positives = len(set(predicted_outliers) & set(true_outliers))
    
    precision = true_positives / top_k_eff if top_k_eff > 0 else 0.0
    recall = true_positives / n_true_outliers if n_true_outliers > 0 else 0.0
    true_positive_rate = true_positives / n_true_outliers if n_true_outliers > 0 else 0.0
    
    return precision, recall, true_positive_rate


@dataclass
class OutlierResults:
    dataset: str
    n_train: int
    n_test: int
    n_outliers: int
    dim: int
    outlier_type: str
    outlier_ratio: float
    top_k: int
    
    endtree_build_time_s: float
    endtree_query_time_ms: float
    endtree_precision: float
    endtree_recall: float
    endtree_tpr: float
    
    lof_fit_time_s: float
    lof_query_time_ms: float
    lof_precision: float
    lof_recall: float
    lof_tpr: float


def run_outlier_experiment(
    *,
    dataset: str,
    n_train: int,
    n_test: int,
    seed: int,
    outlier_ratio: float,
    outlier_type: str,
    top_k: int,
    ef: int,
    M: int,
    best: bool,
    glove_dim: int = 100,
    random_dim: int = 128,
) -> OutlierResults:
    """Run outlier detection comparison on a single dataset."""
    rng = np.random.default_rng(seed)
    ds = dataset.strip().lower()
    
    # --- Load dataset ---
    if ds == "mnist":
        x_train_u8, y_train_u8, x_test_u8, _y_test_u8 = load_mnist()
        n_train = min(n_train, x_train_u8.shape[0])
        n_test = min(n_test, x_test_u8.shape[0])
        train_idx = rng.choice(x_train_u8.shape[0], size=n_train, replace=False)
        test_idx = rng.choice(x_test_u8.shape[0], size=n_test, replace=False)
        x_train = (x_train_u8[train_idx].astype(np.float32) / 255.0).astype(np.float32, copy=False)
        x_test = (x_test_u8[test_idx].astype(np.float32) / 255.0).astype(np.float32, copy=False)
    elif ds == "glove":
        x_train, x_test = load_glove(n_train=n_train, n_query=n_test, dim=glove_dim)
    elif ds == "sift":
        x_train, x_test = load_sift(n_train=n_train, n_query=n_test)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    dim = x_train.shape[1]
    
    # --- Inject outliers into test set ---
    test_data, test_labels = inject_outliers(
        x_train=x_train,
        x_test=x_test,
        outlier_ratio=outlier_ratio,
        outlier_type=outlier_type,
        rng=rng,
    )
    n_outliers = int(np.sum(test_labels))
    
    print(f"\n=== Dataset: {dataset.upper()} ===")
    print(f"Training data: {x_train.shape[0]} clean samples")
    print(f"Test data: {test_data.shape[0]} samples ({n_outliers} outliers, {test_data.shape[0] - n_outliers} inliers)")
    print(f"Dimensionality: {dim}")
    print(f"Outlier type: {outlier_type}, ratio: {outlier_ratio:.2%}")
    print()
    
    # --- Build EndTree on clean training data ---
    print("Building EndTree on clean training data...")
    print(f"  Using M={M} (expected layers: ~{int(np.log2(x_train.shape[0]) / M)}), best={best}, ef={ef}")
    et = hnswlib.EndTree(space="l2", dim=dim)
    t0 = time.perf_counter()
    et.build(x_train, M=M, best=best)
    t1 = time.perf_counter()
    endtree_build_time = t1 - t0
    print(f"EndTree built in {endtree_build_time:.4f}s")
    
    # --- Calculate EndTree outlier scores ---
    print("Calculating EndTree outlier scores...")
    t0 = time.perf_counter()
    endtree_scores = np.array([
        float(et.outlier_score_path(test_data[i], ef=ef))
        for i in range(len(test_data))
    ])
    t1 = time.perf_counter()
    endtree_query_time = (t1 - t0) / len(test_data) * 1000.0  # ms per query
    
    # --- Calculate LOF outlier scores ---
    print("Calculating LOF outlier scores...")
    lof = LocalOutlierFactor(
        n_neighbors=min(20, x_train.shape[0] - 1),
        metric="euclidean",
        novelty=True
    )
    t0 = time.perf_counter()
    lof.fit(x_train)
    t1 = time.perf_counter()
    lof_fit_time = t1 - t0
    
    t0 = time.perf_counter()
    lof_scores = -lof.score_samples(test_data)  # Higher = more outlier
    t1 = time.perf_counter()
    lof_query_time = (t1 - t0) / len(test_data) * 1000.0  # ms per query
    
    # --- Calculate metrics ---
    top_k_eff = min(top_k, n_outliers)
    
    et_precision, et_recall, et_tpr = calculate_metrics(
        endtree_scores, test_labels, top_k_eff
    )
    lof_precision, lof_recall, lof_tpr = calculate_metrics(
        lof_scores, test_labels, top_k_eff
    )
    
    # --- Print results ---
    print(f"\nEndTree Results:")
    print(f"  Build time: {endtree_build_time:.4f}s")
    print(f"  Query time: {endtree_query_time:.4f}ms per sample")
    print(f"  Precision@{top_k_eff}: {et_precision:.4f}")
    print(f"  Recall@{top_k_eff}: {et_recall:.4f}")
    print(f"  True Positive Rate: {et_tpr:.4f}")
    
    print(f"\nLOF Results:")
    print(f"  Fit time: {lof_fit_time:.4f}s")
    print(f"  Query time: {lof_query_time:.4f}ms per sample")
    print(f"  Precision@{top_k_eff}: {lof_precision:.4f}")
    print(f"  Recall@{top_k_eff}: {lof_recall:.4f}")
    print(f"  True Positive Rate: {lof_tpr:.4f}")
    
    print(f"\nScore Statistics:")
    print(f"  EndTree - Min: {endtree_scores.min():.4e}, Max: {endtree_scores.max():.4e}, Mean: {endtree_scores.mean():.4e}")
    print(f"  LOF     - Min: {lof_scores.min():.4e}, Max: {lof_scores.max():.4e}, Mean: {lof_scores.mean():.4e}")
    
    # Separate stats for outliers vs inliers
    outlier_mask = test_labels == 1
    inlier_mask = test_labels == 0
    
    if np.any(outlier_mask):
        print(f"\n  EndTree outlier mean: {endtree_scores[outlier_mask].mean():.4e}")
        print(f"  EndTree inlier mean:  {endtree_scores[inlier_mask].mean():.4e}")
        print(f"  LOF outlier mean: {lof_scores[outlier_mask].mean():.4e}")
        print(f"  LOF inlier mean:  {lof_scores[inlier_mask].mean():.4e}")
    
    return OutlierResults(
        dataset=dataset,
        n_train=x_train.shape[0],
        n_test=test_data.shape[0],
        n_outliers=n_outliers,
        dim=dim,
        outlier_type=outlier_type,
        outlier_ratio=outlier_ratio,
        top_k=top_k_eff,
        endtree_build_time_s=endtree_build_time,
        endtree_query_time_ms=endtree_query_time,
        endtree_precision=et_precision,
        endtree_recall=et_recall,
        endtree_tpr=et_tpr,
        lof_fit_time_s=lof_fit_time,
        lof_query_time_ms=lof_query_time,
        lof_precision=lof_precision,
        lof_recall=lof_recall,
        lof_tpr=lof_tpr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Outlier detection comparison: EndTree vs LOF"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "glove", "sift", "random", "all"],
        help="Dataset to test (or 'all' to test all datasets)",
    )
    parser.add_argument("--n-train", type=int, default=5000, help="Number of training samples (clean)")
    parser.add_argument("--n-test", type=int, default=1000, help="Number of test samples (before outlier injection)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--outlier-ratio",
        type=float,
        default=0.15,
        help="Fraction of test set to be outliers (0.0-1.0)",
    )
    parser.add_argument(
        "--outlier-type",
        type=str,
        default="uniform",
        choices=["uniform", "gaussian"],
        help="Type of synthetic outliers to inject",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Top-k outliers to evaluate (default: number of true outliers)")
    parser.add_argument("--ef", type=int, default=64, help="EndTree search ef parameter (higher=better quality, slower)")
    parser.add_argument("--M", type=int, default=1, help="EndTree M parameter (controls hierarchy depth: layers=log2(N)/M)")
    parser.add_argument("--no-best", dest="best", action="store_false", help="Use worst layers (default: use best layers)")
    parser.set_defaults(best=True)
    parser.add_argument("--glove-dim", type=int, default=100, choices=[50, 100, 200, 300])
    parser.add_argument("--random-dim", type=int, default=128, help="Dimensionality for random dataset")
    
    args = parser.parse_args()
    
    # Determine datasets to test
    if args.dataset == "all":
        datasets = ["mnist", "glove", "sift", "random"]
    else:
        datasets = [args.dataset]
    
    # Run experiments
    all_results = []
    for ds in datasets:
        # Determine top_k
        n_test_eff = args.n_test
        n_outliers = int(n_test_eff * args.outlier_ratio)
        top_k = args.top_k if args.top_k is not None else n_outliers
        
        try:
            result = run_outlier_experiment(
                dataset=ds,
                n_train=args.n_train,
                n_test=args.n_test,
                seed=args.seed,
                outlier_ratio=args.outlier_ratio,
                outlier_type=args.outlier_type,
                top_k=top_k,
                ef=args.ef,
                M=args.M,
                best=args.best,
                glove_dim=args.glove_dim,
                random_dim=args.random_dim,
            )
            all_results.append(result)
        except Exception as e:
            print(f"\nError running experiment on {ds}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # --- Summary Table ---
    if len(all_results) > 1:
        print("\n" + "="*100)
        print("SUMMARY: EndTree vs LOF Outlier Detection Comparison")
        print("="*100)
        print(f"{'Dataset':<10} {'Method':<8} {'Build/Fit(s)':<12} {'Query(ms)':<12} {'Precision':<10} {'Recall':<10} {'TPR':<10}")
        print("-"*100)
        
        for res in all_results:
            print(f"{res.dataset:<10} {'EndTree':<8} {res.endtree_build_time_s:<12.4f} {res.endtree_query_time_ms:<12.4f} "
                  f"{res.endtree_precision:<10.4f} {res.endtree_recall:<10.4f} {res.endtree_tpr:<10.4f}")
            print(f"{'':<10} {'LOF':<8} {res.lof_fit_time_s:<12.4f} {res.lof_query_time_ms:<12.4f} "
                  f"{res.lof_precision:<10.4f} {res.lof_recall:<10.4f} {res.lof_tpr:<10.4f}")
            print("-"*100)
        
        # Calculate averages
        avg_et_precision = np.mean([r.endtree_precision for r in all_results])
        avg_et_recall = np.mean([r.endtree_recall for r in all_results])
        avg_et_tpr = np.mean([r.endtree_tpr for r in all_results])
        
        avg_lof_precision = np.mean([r.lof_precision for r in all_results])
        avg_lof_recall = np.mean([r.lof_recall for r in all_results])
        avg_lof_tpr = np.mean([r.lof_tpr for r in all_results])
        
        print(f"{'AVERAGE':<10} {'EndTree':<8} {'':<12} {'':<12} "
              f"{avg_et_precision:<10.4f} {avg_et_recall:<10.4f} {avg_et_tpr:<10.4f}")
        print(f"{'AVERAGE':<10} {'LOF':<8} {'':<12} {'':<12} "
              f"{avg_lof_precision:<10.4f} {avg_lof_recall:<10.4f} {avg_lof_tpr:<10.4f}")
        print("="*100)


if __name__ == "__main__":
    main()
