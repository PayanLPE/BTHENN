"""EndTree 2D visualization demo with outlier score color-coding.

Run:
  python3 -m tests.python.endtree_demo

Builds EndTree on clean cluster data (inliers only), then queries with both
outlier points and inlier samples to compare outlier scores.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.neighbors import LocalOutlierFactor

import hnswlib


def main():
    rng = np.random.RandomState(42)
    
    # Create 2D cluster data (INLIERS ONLY - used to build EndTree)
    dim = 2
    n_cluster1 = 30
    n_cluster2 = 30
    n_cluster3 = 25
    
    # Three tight clusters for training data
    cluster1 = rng.normal(loc=[2.0, 2.0], scale=0.3, size=(n_cluster1, dim))
    cluster2 = rng.normal(loc=[5.0, 1.5], scale=0.25, size=(n_cluster2, dim))
    cluster3 = rng.normal(loc=[3.5, 4.5], scale=0.35, size=(n_cluster3, dim))
    
    # Training data: ONLY clusters, no outliers
    train_data = np.vstack([cluster1, cluster2, cluster3]).astype(np.float32)
    n_train = len(train_data)
    
    # Create QUERY points: some outliers + some inlier samples for comparison
    n_outliers = 15
    n_inlier_queries = 15  # sample inliers to query for comparison
    
    # Outlier query points (scattered outside clusters)
    outlier_queries = rng.uniform(low=[-2, -2], high=[9, 8], size=(n_outliers, dim)).astype(np.float32)
    
    # Inlier query points (sampled from cluster distributions)
    inlier_queries = np.vstack([
        rng.normal(loc=[2.0, 2.0], scale=0.3, size=(5, dim)),
        rng.normal(loc=[5.0, 1.5], scale=0.25, size=(5, dim)),
        rng.normal(loc=[3.5, 4.5], scale=0.35, size=(5, dim)),
    ]).astype(np.float32)
    
    # Combine query points
    query_data = np.vstack([outlier_queries, inlier_queries]).astype(np.float32)
    n_queries = len(query_data)
    
    # Track which query points are true outliers (first n_outliers)
    outlier_indices = set(range(n_outliers))
    
    print(f"Training data: {n_train} points (clusters only, NO outliers)")
    print(f"  Cluster 1: {n_cluster1} points around (2.0, 2.0)")
    print(f"  Cluster 2: {n_cluster2} points around (5.0, 1.5)")
    print(f"  Cluster 3: {n_cluster3} points around (3.5, 4.5)")
    print()
    print(f"Query points: {n_queries} total")
    print(f"  Outliers : {n_outliers} scattered points (to detect)")
    print(f"  Inliers  : {n_inlier_queries} points from cluster distributions (for comparison)")
    print()
    
    # Build EndTree on CLEAN data only
    print("Building EndTree on clean cluster data...")
    et = hnswlib.EndTree(space="l2", dim=dim)
    et.build(train_data, M=1, best=True)
    print("EndTree built successfully.")
    print()
    
    # Calculate EndTree outlier scores for QUERY points
    print("Calculating EndTree outlier scores for query points...")
    et_scores = np.array([float(et.outlier_score_path(query_data[i], ef=64)) for i in range(n_queries)])
    
    # Calculate LOF outlier scores (fit on train data, score query data)
    print("Calculating LOF outlier scores for query points...")
    lof = LocalOutlierFactor(n_neighbors=min(20, n_train - 1), metric="euclidean", novelty=True)
    lof.fit(train_data)
    lof_scores = -lof.score_samples(query_data)  # Higher = more outlier
    
    # Normalize scores for color mapping (use log scale for better visualization)
    et_scores_log = np.log1p(et_scores)  # log(1 + score) to handle zeros
    et_scores_norm = (et_scores_log - et_scores_log.min()) / (et_scores_log.max() - et_scores_log.min() + 1e-10)
    
    lof_scores_norm = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min() + 1e-10)
    
    print(f"EndTree score statistics (query points):")
    print(f"  Min  : {et_scores.min():.4e}")
    print(f"  Max  : {et_scores.max():.4e}")
    print(f"  Mean : {et_scores.mean():.4e}")
    print(f"  Median: {np.median(et_scores):.4e}")
    print(f"  Outlier mean: {et_scores[:n_outliers].mean():.4e}")
    print(f"  Inlier mean : {et_scores[n_outliers:].mean():.4e}")
    print()
    
    print(f"LOF score statistics (query points):")
    print(f"  Min  : {lof_scores.min():.4e}")
    print(f"  Max  : {lof_scores.max():.4e}")
    print(f"  Mean : {lof_scores.mean():.4e}")
    print(f"  Median: {np.median(lof_scores):.4e}")
    print(f"  Outlier mean: {lof_scores[:n_outliers].mean():.4e}")
    print(f"  Inlier mean : {lof_scores[n_outliers:].mean():.4e}")
    print()
    
    # Find top outliers by score
    top_k = 15
    et_top_indices = np.argsort(et_scores)[-top_k:][::-1]
    lof_top_indices = np.argsort(lof_scores)[-top_k:][::-1]
    
    et_true_positives = len(outlier_indices.intersection(set(et_top_indices)))
    lof_true_positives = len(outlier_indices.intersection(set(lof_top_indices)))
    
    print(f"Top {top_k} outliers by EndTree score:")
    for rank, idx in enumerate(et_top_indices, 1):
        is_true = "*OUTLIER" if idx in outlier_indices else " inlier"
        print(f"  {rank:2}. Query {idx:3} {is_true}  score={et_scores[idx]:12.4e}  pos=({query_data[idx, 0]:.2f}, {query_data[idx, 1]:.2f})")
    print(f"EndTree true positive rate: {et_true_positives}/{n_outliers} outliers in top-{top_k}")
    print()
    
    print(f"Top {top_k} outliers by LOF score:")
    for rank, idx in enumerate(lof_top_indices, 1):
        is_true = "*OUTLIER" if idx in outlier_indices else " inlier"
        print(f"  {rank:2}. Query {idx:3} {is_true}  score={lof_scores[idx]:12.4e}  pos=({query_data[idx, 0]:.2f}, {query_data[idx, 1]:.2f})")
    print(f"LOF true positive rate: {lof_true_positives}/{n_outliers} outliers in top-{top_k}")
    print()
    
    # Create visualization
    print("Creating visualization...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Left plot: Ground truth - training data (gray) + query points (blue=inlier, red=outlier)
    ax1.scatter(train_data[:, 0], train_data[:, 1], c='lightgray', alpha=0.5, s=30, 
                edgecolors='gray', linewidths=0.3, label='Training data (clusters)')
    query_colors = ['red' if i in outlier_indices else 'blue' for i in range(n_queries)]
    ax1.scatter(query_data[:, 0], query_data[:, 1], c=query_colors, alpha=0.8, s=80, 
                edgecolors='black', linewidths=0.5)
    # Add legend markers
    ax1.scatter([], [], c='red', s=80, label='Query: Outlier')
    ax1.scatter([], [], c='blue', s=80, label='Query: Inlier')
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.set_title('Ground Truth\n(Training=Gray, Query: Red=Outlier, Blue=Inlier)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right', fontsize=9)
    
    # Middle plot: EndTree outlier scores - training data (gray) + query points (color-coded by score)
    ax2.scatter(train_data[:, 0], train_data[:, 1], c='lightgray', alpha=0.5, s=30, 
                edgecolors='gray', linewidths=0.3, label='Training data')
    scatter2 = ax2.scatter(
        query_data[:, 0], query_data[:, 1],
        c=et_scores_norm,
        cmap='YlOrRd',  # Yellow-Orange-Red colormap
        alpha=0.8,
        s=100,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Mark true outliers with circle outline
    true_outlier_data = query_data[:n_outliers]
    ax2.scatter(true_outlier_data[:, 0], true_outlier_data[:, 1], 
                marker='o', s=150, facecolors='none', edgecolors='darkred', linewidths=2, 
                label='True outliers', zorder=10)
    
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.set_title(f'EndTree Outlier Scores\n(Detected: {et_true_positives}/{n_outliers})', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.legend(loc='upper right', fontsize=9)
    
    # Add colorbar for EndTree
    cbar2 = plt.colorbar(scatter2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Normalized Outlier Score', rotation=270, labelpad=20, fontsize=11)
    
    # Right plot: LOF outlier scores - training data (gray) + query points (color-coded by score)
    ax3.scatter(train_data[:, 0], train_data[:, 1], c='lightgray', alpha=0.5, s=30, 
                edgecolors='gray', linewidths=0.3, label='Training data')
    scatter3 = ax3.scatter(
        query_data[:, 0], query_data[:, 1],
        c=lof_scores_norm,
        cmap='YlOrRd',  # Yellow-Orange-Red colormap
        alpha=0.8,
        s=100,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Mark true outliers with circle outline
    ax3.scatter(true_outlier_data[:, 0], true_outlier_data[:, 1], 
                marker='o', s=150, facecolors='none', edgecolors='darkred', linewidths=2, 
                label='True outliers', zorder=10)
    
    ax3.set_xlabel('X', fontsize=12)
    ax3.set_ylabel('Y', fontsize=12)
    ax3.set_title(f'LOF Outlier Scores\n(Detected: {lof_true_positives}/{n_outliers})', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    ax3.legend(loc='upper right', fontsize=9)
    
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
