"""EndTree 2D visualization demo with outlier score color-coding.

Run:
  python3 -m tests.python.endtree_demo

Creates 100 points in 2D (clusters + outliers), builds EndTree and LOF, and visualizes
each point colored by its outlier score for comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.neighbors import LocalOutlierFactor

import hnswlib


def main():
    rng = np.random.RandomState(42)
    
    # Create 2D data with clusters and outliers
    dim = 2
    n_cluster1 = 30
    n_cluster2 = 30
    n_cluster3 = 25
    n_outliers = 15
    
    # Three tight clusters
    cluster1 = rng.normal(loc=[2.0, 2.0], scale=0.3, size=(n_cluster1, dim))
    cluster2 = rng.normal(loc=[5.0, 1.5], scale=0.25, size=(n_cluster2, dim))
    cluster3 = rng.normal(loc=[3.5, 4.5], scale=0.35, size=(n_cluster3, dim))
    
    # Scattered outliers
    # outliers = rng.uniform(low=[0.0, 0.0], high=[7.0, 6.0], size=(n_outliers, dim))
    outliers = rng.uniform(low=[-2, -2], high=[9, 8], size=(n_outliers, dim))
    
    # Combine all points
    data = np.vstack([cluster1, cluster2, cluster3, outliers]).astype(np.float32)
    n_total = len(data)
    
    # Track which points are true outliers
    outlier_indices = set(range(n_cluster1 + n_cluster2 + n_cluster3, n_total))
    
    print(f"Created {n_total} points in 2D:")
    print(f"  Cluster 1: {n_cluster1} points around (2.0, 2.0)")
    print(f"  Cluster 2: {n_cluster2} points around (5.0, 1.5)")
    print(f"  Cluster 3: {n_cluster3} points around (3.5, 4.5)")
    print(f"  Outliers : {n_outliers} scattered points")
    print()
    
    # Build EndTree
    print("Building EndTree...")
    et = hnswlib.EndTree(space="l2", dim=dim)
    et.build(data, M=1, best=True)
    print("EndTree built successfully.")
    print()
    
    # Calculate EndTree outlier scores for all points
    print("Calculating EndTree outlier scores...")
    et_scores = np.array([float(et.outlier_score_path(data[i], ef=64)) for i in range(n_total)])
    
    # Calculate LOF outlier scores
    print("Calculating LOF outlier scores...")
    lof = LocalOutlierFactor(n_neighbors=min(20, n_total - 1), metric="euclidean")
    lof.fit(data)
    lof_scores = -lof.negative_outlier_factor_  # Higher = more outlier
    
    # Normalize scores for color mapping (use log scale for better visualization)
    et_scores_log = np.log1p(et_scores)  # log(1 + score) to handle zeros
    et_scores_norm = (et_scores_log - et_scores_log.min()) / (et_scores_log.max() - et_scores_log.min() + 1e-10)
    
    lof_scores_norm = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min() + 1e-10)
    
    print(f"EndTree score statistics:")
    print(f"  Min  : {et_scores.min():.4e}")
    print(f"  Max  : {et_scores.max():.4e}")
    print(f"  Mean : {et_scores.mean():.4e}")
    print(f"  Median: {np.median(et_scores):.4e}")
    print()
    
    print(f"LOF score statistics:")
    print(f"  Min  : {lof_scores.min():.4e}")
    print(f"  Max  : {lof_scores.max():.4e}")
    print(f"  Mean : {lof_scores.mean():.4e}")
    print(f"  Median: {np.median(lof_scores):.4e}")
    print()
    
    # Find top outliers by score
    top_k = 15
    et_top_indices = np.argsort(et_scores)[-top_k:][::-1]
    lof_top_indices = np.argsort(lof_scores)[-top_k:][::-1]
    
    et_true_positives = len(outlier_indices.intersection(set(et_top_indices)))
    lof_true_positives = len(outlier_indices.intersection(set(lof_top_indices)))
    
    print(f"Top {top_k} outliers by EndTree score:")
    for rank, idx in enumerate(et_top_indices, 1):
        is_true = "*" if idx in outlier_indices else " "
        print(f"  {rank:2}. Point {idx:3}{is_true}  score={et_scores[idx]:12.4e}  pos=({data[idx, 0]:.2f}, {data[idx, 1]:.2f})")
    print(f"EndTree true positive rate in top-{top_k}: {et_true_positives}/{n_outliers} outliers detected")
    print()
    
    print(f"Top {top_k} outliers by LOF score:")
    for rank, idx in enumerate(lof_top_indices, 1):
        is_true = "*" if idx in outlier_indices else " "
        print(f"  {rank:2}. Point {idx:3}{is_true}  score={lof_scores[idx]:12.4e}  pos=({data[idx, 0]:.2f}, {data[idx, 1]:.2f})")
    print(f"LOF true positive rate in top-{top_k}: {lof_true_positives}/{n_outliers} outliers detected")
    print()
    
    # Create visualization
    print("Creating visualization...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Left plot: Ground truth (clusters vs outliers)
    colors_true = ['blue' if i not in outlier_indices else 'red' for i in range(n_total)]
    ax1.scatter(data[:, 0], data[:, 1], c=colors_true, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.set_title('Ground Truth\n(Blue=Cluster, Red=Outlier)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Middle plot: EndTree outlier scores (color-coded)
    scatter2 = ax2.scatter(
        data[:, 0], data[:, 1],
        c=et_scores_norm,
        cmap='YlOrRd',  # Yellow-Orange-Red colormap
        alpha=0.7,
        s=80,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Highlight top outliers with markers
    et_top_data = data[et_top_indices]
    ax2.scatter(et_top_data[:, 0], et_top_data[:, 1], 
                marker='x', s=200, c='darkred', linewidths=2, 
                label=f'Top {top_k} outliers', zorder=10)
    
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.set_title(f'EndTree Outlier Scores\n(Detected: {et_true_positives}/{n_outliers})', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.legend(loc='upper right', fontsize=10)
    
    # Add colorbar for EndTree
    cbar2 = plt.colorbar(scatter2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Normalized Outlier Score', rotation=270, labelpad=20, fontsize=11)
    
    # Right plot: LOF outlier scores (color-coded)
    scatter3 = ax3.scatter(
        data[:, 0], data[:, 1],
        c=lof_scores_norm,
        cmap='YlOrRd',  # Yellow-Orange-Red colormap
        alpha=0.7,
        s=80,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Highlight top outliers with markers
    lof_top_data = data[lof_top_indices]
    ax3.scatter(lof_top_data[:, 0], lof_top_data[:, 1], 
                marker='x', s=200, c='darkred', linewidths=2, 
                label=f'Top {top_k} outliers', zorder=10)
    
    ax3.set_xlabel('X', fontsize=12)
    ax3.set_ylabel('Y', fontsize=12)
    ax3.set_title(f'LOF Outlier Scores\n(Detected: {lof_true_positives}/{n_outliers})', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    ax3.legend(loc='upper right', fontsize=10)
    
    # Add colorbar for LOF
    cbar3 = plt.colorbar(scatter3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('Normalized Outlier Score', rotation=270, labelpad=20, fontsize=11)
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'endtree_2d_outlier_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Show plot
    plt.show()
    print("\nDemo complete!")


if __name__ == "__main__":
    main()
