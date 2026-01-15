import csv
from pathlib import Path
from experiment import run_experiment
from dataclasses import asdict


def sweep_range_width_to_csv(
    output_path,
    range_widths,
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

if __name__ == "__main__":
    # Example: sweep range_width = 5, 6, 7, 8, 9, 10
    widths = [5, 6, 7, 8, 9, 10]

    sweep_range_width_to_csv(
        output_path=Path("results_mnist_id_range_width.csv"),
        range_widths=widths,
        n_train=10000,
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
