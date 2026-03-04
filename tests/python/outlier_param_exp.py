#!/usr/bin/env python3
"""Parameter sweep experiment for EndTree outlier detection.

Tests how M, ef, n_train, and outlier_ratio affect outlier detection accuracy.
Generates visualizations comparing EndTree vs LOF across parameter ranges.

Run:
  python3 -m tests.python.outlier_param_exp --sweep M
  python3 -m tests.python.outlier_param_exp --sweep ef
  python3 -m tests.python.outlier_param_exp --sweep n_train
  python3 -m tests.python.outlier_param_exp --sweep all
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from outlier_experiment import run_outlier_experiment, OutlierResults


def sweep_parameter(
    param_name: str,
    param_values: List[Any],
    output_dir: Path,
    dataset: str = "mnist",
    n_train: int = 5000,
    n_test: int = 1000,
    seed: int = 42,
    outlier_ratio: float = 0.15,
    outlier_type: str = "uniform",
    ef: int = 200,
    M: int = 4,
    best: bool = True,
    glove_dim: int = 100,
    random_dim: int = 128,
) -> pd.DataFrame:
    """Sweep a single parameter and collect results."""
    
    rows = []
    for val in param_values:
        print(f"\n{'='*60}")
        print(f"[sweep] {param_name}={val}")
        print(f"{'='*60}")
        
        # Build kwargs with the swept parameter
        kwargs = dict(
            dataset=dataset,
            n_train=n_train,
            n_test=n_test,
            seed=seed,
            outlier_ratio=outlier_ratio,
            outlier_type=outlier_type,
            top_k=int(n_test * outlier_ratio),
            ef=ef,
            M=M,
            best=best,
            glove_dim=glove_dim,
            random_dim=random_dim,
        )
        
        # Override the swept parameter
        if param_name == "M":
            kwargs["M"] = val
        elif param_name == "ef":
            kwargs["ef"] = val
        elif param_name == "n_train":
            kwargs["n_train"] = val
        elif param_name == "outlier_ratio":
            kwargs["outlier_ratio"] = val
            kwargs["top_k"] = int(n_test * val)
        
        try:
            res = run_outlier_experiment(**kwargs)
            row = asdict(res)
            row[param_name] = val  # Ensure swept param is in row
            rows.append(row)
        except Exception as e:
            print(f"[sweep] Error at {param_name}={val}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not rows:
        raise RuntimeError(f"No results collected for {param_name} sweep")
    
    # Save to CSV
    df = pd.DataFrame(rows)
    output_path = output_dir / f"sweep_{param_name}_{dataset}.csv"
    df.to_csv(output_path, index=False)
    print(f"\n[sweep] Saved {len(rows)} rows to {output_path}")
    
    return df


def plot_sweep_results(
    df: pd.DataFrame,
    param_name: str,
    output_dir: Path,
    dataset: str,
) -> None:
    """Create visualization for parameter sweep results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"EndTree vs LOF: {param_name} Sweep on {dataset.upper()}", fontsize=14, fontweight='bold')
    
    x = df[param_name]
    
    # Plot 1: Precision comparison
    ax1 = axes[0, 0]
    ax1.plot(x, df['endtree_precision'], marker='o', linewidth=2, markersize=8, label='EndTree', color='#2196F3')
    ax1.plot(x, df['lof_precision'], marker='s', linewidth=2, markersize=8, label='LOF', color='#FF5722')
    ax1.set_xlabel(param_name, fontsize=11)
    ax1.set_ylabel('Precision', fontsize=11)
    ax1.set_title('Precision vs ' + param_name, fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Plot 2: Recall comparison
    ax2 = axes[0, 1]
    ax2.plot(x, df['endtree_recall'], marker='o', linewidth=2, markersize=8, label='EndTree', color='#2196F3')
    ax2.plot(x, df['lof_recall'], marker='s', linewidth=2, markersize=8, label='LOF', color='#FF5722')
    ax2.set_xlabel(param_name, fontsize=11)
    ax2.set_ylabel('Recall', fontsize=11)
    ax2.set_title('Recall vs ' + param_name, fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    # Plot 3: Query time comparison
    ax3 = axes[1, 0]
    ax3.plot(x, df['endtree_query_time_ms'], marker='o', linewidth=2, markersize=8, label='EndTree', color='#2196F3')
    ax3.plot(x, df['lof_query_time_ms'], marker='s', linewidth=2, markersize=8, label='LOF', color='#FF5722')
    ax3.set_xlabel(param_name, fontsize=11)
    ax3.set_ylabel('Query Time (ms/sample)', fontsize=11)
    ax3.set_title('Query Time vs ' + param_name, fontsize=12)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Build/Fit time comparison
    ax4 = axes[1, 1]
    ax4.plot(x, df['endtree_build_time_s'], marker='o', linewidth=2, markersize=8, label='EndTree Build', color='#2196F3')
    ax4.plot(x, df['lof_fit_time_s'], marker='s', linewidth=2, markersize=8, label='LOF Fit', color='#FF5722')
    ax4.set_xlabel(param_name, fontsize=11)
    ax4.set_ylabel('Build/Fit Time (s)', fontsize=11)
    ax4.set_title('Build Time vs ' + param_name, fontsize=12)
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / f"sweep_{param_name}_{dataset}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[plot] Saved figure to {output_path}")
    plt.close()


def plot_combined_accuracy(
    results: Dict[str, pd.DataFrame],
    output_dir: Path,
    dataset: str,
) -> None:
    """Create a combined plot showing accuracy across all parameter sweeps."""
    
    n_params = len(results)
    if n_params == 0:
        return
    
    fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 5))
    if n_params == 1:
        axes = [axes]
    
    fig.suptitle(f"EndTree Precision Across Parameter Sweeps ({dataset.upper()})", fontsize=14, fontweight='bold')
    
    for ax, (param_name, df) in zip(axes, results.items()):
        x = df[param_name]
        ax.plot(x, df['endtree_precision'], marker='o', linewidth=2, markersize=8, label='EndTree', color='#2196F3')
        ax.plot(x, df['lof_precision'], marker='s', linewidth=2, markersize=8, label='LOF', color='#FF5722')
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect')
        ax.set_xlabel(param_name, fontsize=11)
        ax.set_ylabel('Precision', fontsize=11)
        ax.set_title(f'{param_name} Sweep', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    output_path = output_dir / f"combined_accuracy_{dataset}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[plot] Saved combined figure to {output_path}")
    plt.close()


def run_all_sweeps(
    output_dir: Path,
    dataset: str = "mnist",
    n_train: int = 5000,
    n_test: int = 1000,
    seed: int = 42,
    outlier_ratio: float = 0.15,
    outlier_type: str = "uniform",
) -> Dict[str, pd.DataFrame]:
    """Run all parameter sweeps and generate plots."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    
    # Sweep M: hierarchy depth control
    # M controls layers = log2(N) / M
    # For n_train=5000: M=1 → 12 layers, M=2 → 6 layers, M=4 → 3 layers, M=6 → 2 layers
    print("\n" + "="*80)
    print("SWEEPING M (hierarchy depth)")
    print("="*80)
    M_values = [1, 2, 3, 4, 5, 6, 8]
    df_M = sweep_parameter(
        param_name="M",
        param_values=M_values,
        output_dir=output_dir,
        dataset=dataset,
        n_train=n_train,
        n_test=n_test,
        seed=seed,
        outlier_ratio=outlier_ratio,
        outlier_type=outlier_type,
        ef=200,
        M=8,  # default, will be overridden
        best=True,
    )
    plot_sweep_results(df_M, "M", output_dir, dataset)
    results["M"] = df_M
    
    # Sweep ef: search quality (with M=1 to test effect of ef on deep hierarchy)
    print("\n" + "="*80)
    print("SWEEPING ef (search quality)")
    print("="*80)
    ef_values = [16, 32, 64, 128, 200, 300, 500]
    df_ef = sweep_parameter(
        param_name="ef",
        param_values=ef_values,
        output_dir=output_dir,
        dataset=dataset,
        n_train=n_train,
        n_test=n_test,
        seed=seed,
        outlier_ratio=outlier_ratio,
        outlier_type=outlier_type,
        ef=200,  # default, will be overridden
        M=1,  # Use M=1 to test ef effect on deep hierarchy
        best=True,
    )
    plot_sweep_results(df_ef, "ef", output_dir, dataset)
    results["ef"] = df_ef
    
    # Generate combined plot
    plot_combined_accuracy(results, output_dir, dataset)
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parameter sweep experiment for EndTree outlier detection"
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default="all",
        choices=["M", "ef", "all"],
        help="Parameter to sweep (or 'all' for all parameters)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "glove", "sift"],
        help="Dataset to use",
    )
    parser.add_argument("--n-train", type=int, default=5000, help="Number of training samples")
    parser.add_argument("--n-test", type=int, default=1000, help="Number of test samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outlier-ratio", type=float, default=0.15)
    parser.add_argument("--outlier-type", type=str, default="uniform", choices=["uniform", "gaussian"])
    parser.add_argument("--output-dir", type=str, default="results/outlier_sweeps", help="Output directory")
    
    # Parameters for single sweeps
    parser.add_argument("--ef", type=int, default=200, help="Default ef for non-ef sweeps")
    parser.add_argument("--M", type=int, default=8, help="Default M for non-M sweeps")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.sweep == "all":
        run_all_sweeps(
            output_dir=output_dir,
            dataset=args.dataset,
            n_train=args.n_train,
            n_test=args.n_test,
            seed=args.seed,
            outlier_ratio=args.outlier_ratio,
            outlier_type=args.outlier_type,
        )
    else:
        # Single parameter sweep
        if args.sweep == "M":
            values = [1, 2, 3, 4, 5, 6, 8]
            M_default = args.M
        elif args.sweep == "ef":
            values = [16, 32, 64, 128, 200, 300, 500]
            M_default = 1  # Use M=1 to test ef effect on deep hierarchy
        else:
            raise ValueError(f"Unknown sweep: {args.sweep}")
        
        df = sweep_parameter(
            param_name=args.sweep,
            param_values=values,
            output_dir=output_dir,
            dataset=args.dataset,
            n_train=args.n_train,
            n_test=args.n_test,
            seed=args.seed,
            outlier_ratio=args.outlier_ratio,
            outlier_type=args.outlier_type,
            ef=args.ef,
            M=M_default,
            best=True,
        )
        plot_sweep_results(df, args.sweep, output_dir, args.dataset)
    
    print("\n" + "="*80)
    print("ALL SWEEPS COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
