import csv
from pathlib import Path
from experiment import run_experiment
from dataclasses import asdict
import pandas as pd
import matplotlib.pyplot as plt


def sweep_range_width_to_csv(
    output_path,
    range_widths,
    dataset = "mnist",
    n_train = 10_000,
    n_query = 200,
    k = 10,
    seed = 123,
    label_mode = "id",
    low_high = (0.0, 200.0),
    M = 16,
    ef_construction = 200,
    ef = 200,
    best = True,
) -> None:
    rows = []
    for w in range_widths:
        print(f"[sweep] range_width={w}")
        res = run_experiment(
            dataset=dataset,
            n_train=n_train,
            n_query=n_query,
            k=k,
            seed=seed,
            label_mode=label_mode,
            low_high=low_high,
            range_width=w,
            M=M,
            ef_construction=ef_construction,
            ef=ef,
            best=best,
            glove_dim=128,
        )
        rows.append(asdict(res))

    if not rows:
        raise RuntimeError("No rows recorded in sweep_range_width_to_csv")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[sweep] wrote {len(rows)} rows to {output_path}")


    # Read the data back and create visualizations
    df = pd.read_csv(output_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Query time comparison
    axes[0, 0].plot(df['range_width'], df['avg_query_time_bthenn_ms'], marker='o', label='BTHENN')
    axes[0, 0].plot(df['range_width'], df['avg_query_time_henn_ms'], marker='s', label='HENN')
    axes[0, 0].set_xlabel('Range Width')
    axes[0, 0].set_ylabel('Avg Query Time (ms)')
    axes[0, 0].set_title('Query Time vs Range Width')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Recall comparison
    axes[0, 1].plot(df['range_width'], df['recall_bthenn'], marker='o', label='BTHENN')
    axes[0, 1].plot(df['range_width'], df['recall_henn'], marker='s', label='HENN')
    axes[0, 1].set_xlabel('Range Width')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].set_title('Recall vs Range Width')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.05])
    
    # Plot 3: Build time comparison
    axes[1, 0].plot(df['range_width'], df['build_time_bthenn_s'], marker='o', label='BTHENN')
    axes[1, 0].plot(df['range_width'], df['build_time_henn_s'], marker='s', label='HENN')
    axes[1, 0].set_xlabel('Range Width')
    axes[1, 0].set_ylabel('Build Time (s)')
    axes[1, 0].set_title('Build Time vs Range Width')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Success rates
    axes[1, 1].plot(df['range_width'], df['bthenn_no_result_rate'], marker='o', label='BTHENN No Result Rate')
    axes[1, 1].plot(df['range_width'], df['henn_no_result_rate'], marker='s', label='HENN No Result Rate')
    axes[1, 1].plot(df['range_width'], df['success_rate_bthenn_when_henn_empty'], marker='^', label='BTHENN Success (HENN Empty)')
    axes[1, 1].set_xlabel('Range Width')
    axes[1, 1].set_ylabel('Rate')
    axes[1, 1].set_title('Success/Failure Rates vs Range Width')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path.with_suffix('.png'))
    

if __name__ == "__main__":
    # Example: sweep range_width = 5, 6, 7, 8, 9, 10
    widths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    dataset = "random"
    n_train=20000

    sweep_range_width_to_csv(
        output_path=Path(f"results_{dataset}_id_range_width_{n_train}.csv"),
        range_widths=widths,
        n_train=n_train,
        n_query=200,
        k=10,
        seed=100,
        label_mode="id",
        low_high=(0.0, 200.0),
        M=16,
        ef_construction=100,
        ef=50,
        best=True,
    )
