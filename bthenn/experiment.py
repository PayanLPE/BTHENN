#!/usr/bin/env python3
"""BtHENN vs HENN experiment on MNIST.

This script:
- Downloads/parses MNIST (no sklearn/tensorflow dependency).
- Builds BtHENN (B-tree + HENN hybrid) where the B-tree key is a *quantitative*
	per-vector label.
- Builds standalone HENN (hnswlib with build_henn).
- Runs *range-filtered* ANN queries: only items whose label is within
	[low, high] are eligible.

Label modes:
- "id" (default): each entry gets a unique numeric label 0..n_train-1.
- "digit": label is the MNIST digit (0..9).
- "random": label is a deterministic random float in [0, 1).

Reports:
- Index build time (BtHENN vs HENN)
- Avg query time
- Recall@k within the label range
- Success rate: % queries where BtHENN returns the exact top-1 within-range
	while standalone HENN fails or yields no result.
"""

from __future__ import annotations

import argparse
import gzip
import json
import struct
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

import bthenn
import hnswlib


MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = {
	"train_images": "train-images-idx3-ubyte.gz",
	"train_labels": "train-labels-idx1-ubyte.gz",
	"test_images": "t10k-images-idx3-ubyte.gz",
	"test_labels": "t10k-labels-idx1-ubyte.gz",
}

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
	cache_dir = Path.home() / ".cache" / "bthenn" / "mnist"
	cache_dir.mkdir(parents=True, exist_ok=True)
	paths = {k: cache_dir / v for k, v in MNIST_FILES.items()}
	for v in MNIST_FILES.values():
		url = MNIST_BASE_URL + v
		dest = cache_dir / v
		dest.parent.mkdir(parents=True, exist_ok=True)
		if dest.exists() and dest.stat().st_size > 0:
			continue
		with urllib.request.urlopen(url) as response, open(dest, "wb") as f:
			f.write(response.read())

	x_train_u8 = _read_idx_images_gz(paths["train_images"])
	y_train_u8 = _read_idx_labels_gz(paths["train_labels"])
	x_test_u8 = _read_idx_images_gz(paths["test_images"])
	y_test_u8 = _read_idx_labels_gz(paths["test_labels"])
	return x_train_u8, y_train_u8, x_test_u8, y_test_u8


def _recall_at_k(found: Sequence[int], truth: Sequence[int], k: int) -> float:
	if k <= 0:
		return 0.0
	if not truth:
		return 0.0
	k_eff = min(int(k), len(truth))
	if k_eff <= 0:
		return 0.0
	found_k = list(found)[:k_eff]
	truth_k = list(truth)[:k_eff]
	return len(set(found_k) & set(truth_k)) / float(k_eff)


def _exact_topk_within_subset(
	x_train: np.ndarray,
	x_train_norms: np.ndarray,
	query: np.ndarray,
	subset_idx: np.ndarray,
	k: int,
) -> List[int]:
	if subset_idx.size == 0:
		return []
	q = query.astype(np.float32, copy=False)
	q_norm = float(np.dot(q, q))
	x_sub = x_train[subset_idx]
	d2 = x_train_norms[subset_idx] + q_norm - 2.0 * (x_sub @ q)
	k_eff = min(k, d2.shape[0])
	if k_eff <= 0:
		return []
	part = np.argpartition(d2, kth=k_eff - 1)[:k_eff]
	order = part[np.argsort(d2[part])]
	return subset_idx[order].astype(int).tolist()


def _make_range_filter(labels: np.ndarray, lo: float, hi: float) -> Callable[[int], bool]:
	lo_f = float(lo)
	hi_f = float(hi)

	def _f(idx: int) -> bool:
		v = float(labels[int(idx)])
		return lo_f <= v <= hi_f

	return _f


def _labels_for_mode(*, mode: str, y_train_digits: np.ndarray, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
	mode_l = mode.strip().lower()
	if mode_l == "id":
		return np.arange(y_train_digits.shape[0], dtype=np.float32)
	if mode_l == "random":
		return rng.random(y_train_digits.shape[0], dtype=np.float32) * (high - low) + low


def _pick_range_for_query(
	*,
	labels: np.ndarray,
	rng: np.random.Generator,
	width: float,
) -> Tuple[float, float, int]:
	anchor = int(rng.integers(0, labels.shape[0]))
	center = float(labels[anchor])
	hw = float(width) / 2.0
	lo = center - hw
	hi = center + hw
	# Clamp to global min/max so digit-mode doesn't create empty ranges too often.
	lo = max(lo, float(np.min(labels)))
	hi = min(hi, float(np.max(labels)))
	if lo > hi:
		lo, hi = hi, lo
	return float(lo), float(hi), anchor


@dataclass
class Results:
	n_train: int
	n_query: int
	n_query_evaluated: int
	dim: int
	k: int
	label_mode: str
	range_width: float
	M: int
	ef_construction: int
	ef: int
	build_time_bthenn_s: float
	build_time_henn_s: float
	avg_query_time_bthenn_ms: float
	avg_query_time_henn_ms: float
	recall_bthenn: float
	recall_henn: float
	henn_no_result_rate: float
	bthenn_no_result_rate: float
	success_rate_bthenn_when_henn_empty: float


def run_experiment(
	*,
	n_train: int,
	n_query: int,
	k: int,
	seed: int,
	label_mode: str,
	low_high: Optional[Tuple[float, float]] = None,
	range_width: float,
	M: int,
	ef_construction: int,
	ef: int,
	best: bool,
) -> Results:
	x_train_u8, y_train_u8, x_test_u8, y_test_u8 = load_mnist()
	rng = np.random.default_rng(seed)

	n_train = min(n_train, x_train_u8.shape[0])
	n_query = min(n_query, x_test_u8.shape[0])
	train_idx = rng.choice(x_train_u8.shape[0], size=n_train, replace=False)
	query_idx = rng.choice(x_test_u8.shape[0], size=n_query, replace=False)

	x_train = (x_train_u8[train_idx].astype(np.float32) / 255.0).astype(np.float32, copy=False)
	y_train_int = y_train_u8[train_idx].astype(np.int32)
	labels = _labels_for_mode(mode=label_mode, y_train_digits=y_train_int, low=low_high[0] if low_high is not None else 0, high=low_high[1] if low_high is not None else 200, rng=rng)

	x_query = (x_test_u8[query_idx].astype(np.float32) / 255.0).astype(np.float32, copy=False)
	# Query vectors come from MNIST test set; they don't need labels.

	dim = int(x_train.shape[1])
	x_train_norms = np.sum(x_train * x_train, axis=1).astype(np.float32)

	labels_min = float(np.min(labels))
	labels_max = float(np.max(labels))
	if not np.isfinite(labels_min) or not np.isfinite(labels_max):
		raise ValueError("Non-finite labels encountered")
	if float(range_width) <= 0.0:
		raise ValueError("range_width must be > 0")

	# --- Build BtHENN ---
	tree = bthenn.Index16()
	t0 = time.perf_counter()
	tree.build_btree_henn(x_train, labels, int(x_train.shape[0]), int(x_train.shape[1]))
	t1 = time.perf_counter()
	build_time_bthenn = t1 - t0

	# --- Build standalone HENN (hnswlib) ---
	ids = np.arange(x_train.shape[0])
	p = hnswlib.Index(space="l2", dim=dim)
	p.init_index(max_elements=int(x_train.shape[0]), ef_construction=int(ef_construction), M=int(M))
	t2 = time.perf_counter()
	p.build_henn(x_train, ids=ids, M=int(M), best=bool(best))
	t3 = time.perf_counter()
	build_time_henn = t3 - t2
	p.set_ef(int(ef))

	# --- Query benchmark + metrics ---
	total_qtime_bthenn = 0.0
	total_qtime_henn = 0.0
	recall_sum_bthenn = 0.0
	recall_sum_henn = 0.0

	bthenn_no_result = 0
	henn_no_result = 0
	bthenn_correct_top1 = 0
	henn_correct_top1 = 0
	bthenn_correct_top1_when_henn_empty = 0

	query_evaluated = 0

	for q in x_query:
		lo, hi, _anchor = _pick_range_for_query(labels=labels, rng=rng, width=float(range_width))
		# Determine the in-range subset.
		subset_mask = (labels >= lo) & (labels <= hi)
		subset_idx = np.where(subset_mask)[0].astype(np.int32)
		if subset_idx.size == 0:
			# No points eligible; skip from metrics.
			continue

		truth = _exact_topk_within_subset(x_train, x_train_norms, q, subset_idx, k)
		if not truth:
			continue
		required = len(truth)

		# BtHENN range search
		t_q0 = time.perf_counter()
		try:
			res_b = tree.rangeSearchKNN(float(lo), float(hi), q.astype(np.float32, copy=False), int(k))
		except Exception:
			res_b = []
		t_q1 = time.perf_counter()
		total_qtime_bthenn += (t_q1 - t_q0)
		found_b = [int(idx) for dist, idx in res_b] if res_b else []
		if len(found_b) < required:
			bthenn_no_result += 1

		# HENN filtered search (filter by the same label-range)
		filter_fn = _make_range_filter(labels, lo, hi)
		t_h0 = time.perf_counter()
		try:
			labels_h, _ = p.knn_query(q.reshape(1, -1), k=int(k), num_threads=1, filter=filter_fn)
			found_h = labels_h[0].astype(int).tolist() if labels_h is not None and len(labels_h) else []
			# Some backends can return -1s if not enough results.
			found_h = [v for v in found_h if v >= 0]
		except Exception:
			found_h = []
		t_h1 = time.perf_counter()
		total_qtime_henn += (t_h1 - t_h0)
		henn_failed = len(found_h) < required
		if henn_failed:
			henn_no_result += 1

		# Recall@k (within range)
		recall_sum_bthenn += _recall_at_k(found_b, truth, k)
		recall_sum_henn += _recall_at_k(found_h, truth, k)

		# Top-1 exactness
		gt_top1 = int(truth[0])
		b_ok = bool(found_b) and int(found_b[0]) == gt_top1
		h_ok = bool(found_h) and int(found_h[0]) == gt_top1
		bthenn_correct_top1 += int(b_ok)
		henn_correct_top1 += int(h_ok)

		if b_ok and henn_failed:
			bthenn_correct_top1_when_henn_empty += 1

		query_evaluated += 1

	# NOTE: some queries may be skipped if the sampled range is empty.
	n_query_effective = int(query_evaluated)
	if n_query_effective <= 0:
		raise RuntimeError("No queries evaluated (all sampled ranges empty?)")

	results = Results(
		n_train=int(x_train.shape[0]),
		n_query=int(x_query.shape[0]),
		n_query_evaluated=int(n_query_effective),
		dim=dim,
		k=int(k),
		label_mode=str(label_mode),
		range_width=float(range_width),
		M=int(M),
		ef_construction=int(ef_construction),
		ef=int(ef),
		build_time_bthenn_s=float(build_time_bthenn),
		build_time_henn_s=float(build_time_henn),
		avg_query_time_bthenn_ms=float((total_qtime_bthenn / n_query_effective) * 1000.0),
		avg_query_time_henn_ms=float((total_qtime_henn / n_query_effective) * 1000.0),
		recall_bthenn=float(recall_sum_bthenn / n_query_effective),
		recall_henn=float(recall_sum_henn / n_query_effective),
		henn_no_result_rate=float(henn_no_result / n_query_effective),
		bthenn_no_result_rate=float(bthenn_no_result / n_query_effective),
		success_rate_bthenn_when_henn_empty=float(bthenn_correct_top1_when_henn_empty / n_query_effective),
	)

	return results


def main() -> None:
	parser = argparse.ArgumentParser(description="BtHENN vs HENN benchmark on MNIST")
	parser.add_argument("--n-train", type=int, default=10000)
	parser.add_argument("--n-query", type=int, default=200)
	parser.add_argument("-k", type=int, default=10)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument(
		"--label-mode",
		type=str,
		default="id",
		choices=["id", "random"],
		help="Quantitative label assignment per entry (used for range filtering)",
	)
	parser.add_argument("--low-high", type=float, nargs=2, help="Low and high label values for random label mode")
	parser.add_argument(
		"--range-width",
		type=float,
		default=1000.0,
		help="Width of the numeric label range [lo, hi] sampled per query",
	)
	parser.add_argument("--M", type=int, default=16)
	parser.add_argument("--ef-construction", type=int, default=100)
	parser.add_argument("--ef", type=int, default=50)
	parser.add_argument("--best", action="store_true", help="Use best=True for build_henn")
	args = parser.parse_args()

	results = run_experiment(
		n_train=args.n_train,
		n_query=args.n_query,
		k=args.k,
		seed=args.seed,
		label_mode=args.label_mode,
		low_high=tuple(args.low_high),
		range_width=args.range_width,
		M=args.M,
		ef_construction=args.ef_construction,
		ef=args.ef,
		best=args.best,
	)

	print("=== BtHENN vs HENN (MNIST label-range queries) ===")
	print(
		f"n_train={results.n_train} n_query={results.n_query} evaluated={results.n_query_evaluated} "
		f"dim={results.dim} k={results.k} label_mode={results.label_mode} range_width={results.range_width}"
	)
	print(f"HENN params: M={results.M} ef_construction={results.ef_construction} ef={results.ef}")
	print(f"Build time BtHENN: {results.build_time_bthenn_s:.4f}s")
	print(f"Build time HENN : {results.build_time_henn_s:.4f}s")
	print(f"Avg query BtHENN: {results.avg_query_time_bthenn_ms:.4f}ms")
	print(f"Avg query HENN : {results.avg_query_time_henn_ms:.4f}ms")
	print(f"Recall@{results.k} BtHENN: {results.recall_bthenn:.4f}")
	print(f"Recall@{results.k} HENN : {results.recall_henn:.4f}")
	print(f"No-result rate BtHENN: {results.bthenn_no_result_rate:.4f}")
	print(f"No-result rate HENN : {results.henn_no_result_rate:.4f}")
	print(
		"Success rate (BtHENN returns exact top-1 within-range while HENN yields no result): "
		f"{results.success_rate_bthenn_when_henn_empty:.4f}"
	)


if __name__ == "__main__":
	main()
