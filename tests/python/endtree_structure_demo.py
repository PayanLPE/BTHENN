"""EndTree Structure Demo - Python version of endtree_augmentation_test.cpp

Run:
  python3 -m tests.python.endtree_structure_demo

Demonstrates EndTree outlier scoring on random 2D data.
Note: Python API currently exposes only outlier scores, not internal structure details
like parent-child relationships. For full structural inspection, use the C++ test.
"""

import numpy as np
import hnswlib


def main():
    print("\n=== EndTree Structure Demo with Random Data ===\n")

    # Generate random 2D data
    dim = 2
    n = 50  # More points for better hierarchy
    
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    data = rng.uniform(0.0, 100.0, size=(n, dim)).astype(np.float32)
    
    print(f"Generated {n} random points in {dim}D space")
    print("(showing all points):\n")
    
    for i in range(n):
        print(f"  Point {i:2}: ({data[i, 0]:6.2f}, {data[i, 1]:6.2f})")
    
    # Build EndTree structure
    print("\n--- Building EndTree (HENN + Augmentation) ---\n")
    et = hnswlib.EndTree(space="l2", dim=dim)
    et.build(data, M=1, best=True)
    print("EndTree built successfully.")
    print("  • Hierarchical structure created automatically")
    print("  • Parent-child statistics computed")
    print("  • Ready for anomaly detection")
    
    # Note about structure details
    print("\nNote: Python API currently exposes outlier scores only.")
    print("For detailed structure inspection (layers, parent-child relationships),")
    print("see the C++ test: tests/cpp/endtree_augmentation_test")
    
    # Test various query points
    print("\n--- Outlier Score Testing ---\n")
    
    queries = [
        (50.0, 50.0, "Center"),
        (0.0, 0.0, "Corner"),
        (100.0, 100.0, "Opposite corner"),
        (data[0, 0], data[0, 1], f"On existing point 0"),
        (150.0, 150.0, "Far outlier"),
        (25.0, 75.0, "Random point A"),
        (80.0, 30.0, "Random point B"),
    ]
    
    print(f"Testing {len(queries)} query points:\n")
    
    for i, (x, y, desc) in enumerate(queries, 1):
        q = np.array([x, y], dtype=np.float32)
        score = et.outlier_score_path(q, ef=100)
        
        print(f"  Query {i}: ({x:6.1f}, {y:6.1f}) - {desc}")
        print(f"    Outlier Score: {score:.6e}")
    
    # Score all points in the dataset
    print("\n--- Scoring All Data Points ---\n")
    
    all_scores = np.array([et.outlier_score_path(data[i], ef=100) for i in range(n)])
    
    print(f"Score statistics:")
    print(f"  Min    : {all_scores.min():.6e}")
    print(f"  Max    : {all_scores.max():.6e}")
    print(f"  Mean   : {all_scores.mean():.6e}")
    print(f"  Median : {np.median(all_scores):.6e}")
    print(f"  Std Dev: {all_scores.std():.6e}")
    
    # Find top outliers
    top_k = 10
    top_indices = np.argsort(all_scores)[-top_k:][::-1]
    
    print(f"\nTop {top_k} outliers by score:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank:2}. Point {idx:2}: ({data[idx, 0]:6.2f}, {data[idx, 1]:6.2f})  score={all_scores[idx]:.6e}")
    
    # Find least anomalous points
    bottom_k = 5
    bottom_indices = np.argsort(all_scores)[:bottom_k]
    
    print(f"\nLeast anomalous {bottom_k} points (most normal):")
    for rank, idx in enumerate(bottom_indices, 1):
        print(f"  {rank:2}. Point {idx:2}: ({data[idx, 0]:6.2f}, {data[idx, 1]:6.2f})  score={all_scores[idx]:.6e}")
    
    print("\n=== Summary ===\n")
    print("EndTree Structure:")
    print(f"  • {n} points organized in hierarchical layers")
    print(f"  • Parent-child statistics computed during build")
    print(f"  • Anomaly scores combine distance, density, and structural deviation")
    print(f"  • Scores range from {all_scores.min():.2e} to {all_scores.max():.2e}")
    print("\nFor detailed structural analysis (layers, parent relationships),")
    print("compile and run: ./build/endtree_augmentation_test\n")


if __name__ == "__main__":
    main()
