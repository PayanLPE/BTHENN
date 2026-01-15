#!/usr/bin/env python3
"""
Simple test script for the bthenn Python module
"""

import time
import numpy as np
import bthenn
import hnswlib

def test_basic_functionality():
    print("Testing bthenn module...")
    
    # Create a BTree with float keys, order 5, dimension 2
    tree = bthenn.Index16()
    print("✓ Successfully created BTree instance")

    # Create some sample data for HENN indexing
    data = np.array([[i + j for j in range(2)] for i in range(100)], dtype=np.float32)
    labels = np.array([i for i in range(100)], dtype=np.float32)

    # Build HENN index
    tree.build_btree_henn(data, labels, data.shape[0], data.shape[1])
    print("✓ Successfully built HENN index")

    # Test KNN search on root node
    query = np.array([4.0, 5.0], dtype=np.float32)

    print(data)
    print(labels)
    result = tree.rangeSearchKNN(25, 75, query, 5)
    print(f"Range KNN search result: {result}")
    

def test_range_use_random():
    print("Testing bthenn module with random data and compare with HENN")
    
    for i in range(5):
        # Create some sample data for HENN indexing
        data = np.random.rand(10000, 2).astype(np.float32) * 100
        labels = np.random.randint(0, 10000, size=(10000,)).astype(np.float32)

        query = np.random.rand(2).astype(np.float32) * 100

        # Create a BTree with float keys, order 5, dimension 2
        tree = bthenn.Index16()
        tree.build_btree_henn(data, labels, data.shape[0], data.shape[1])
        # result (distance, index)
        result = tree.rangeSearchKNN(25, 30, query, i+3)
        # Write result to file
        with open('range_knn_results.txt', 'a') as f:
            f.write(f"Query: {query.tolist()}\n")
            f.write(f"K: {i+3}\n")
            f.write(f"Labels: {labels[[idx for dist, idx in result]].tolist()}\n")
            f.write(f"Distances: {[dist for dist, idx in result]}\n")
            f.write("-" * 50 + "\n")
        f.close()

        # verify the distance and labels using brute-force
        dists = np.linalg.norm(data - query, axis=1)
        mask = (labels >= 25) & (labels <= 30)
        filtered_dists = dists[mask]
        filtered_labels = labels[mask]
        if len(filtered_dists) == 0:
            brute_force_result = []
        else:
            knn_indices = np.argsort(filtered_dists)
            brute_force_result = [(filtered_dists[i], filtered_labels[i]) for i in knn_indices]
        # Write brute-force result to file
        with open('brute_force_knn_results.txt', 'a') as f:
            f.write(f"Query: {query.tolist()}\n")
            f.write(f"Labels: {[label for dist, label in brute_force_result]}\n")
            f.write(f"Distances: {[dist for dist, label in brute_force_result]}\n")
            f.write("-" * 50 + "\n")
        f.close()

        # Create a HENN index for comparison
        ids = np.arange(10000)
        p = hnswlib.Index(space='l2', dim=2)
        p.init_index(max_elements=10000, ef_construction=100, M=32)
        p.build_henn(data, ids=ids, M=32, best=True)

        def label_filter(id_val):
            label = labels[int(id_val)] 
            return 25 <= label <= 30

        try:
            indices_henn, distances_henn = p.knn_query(query.reshape(1, -1), k=i+3, filter=label_filter)
        except Exception as e:
            print(f"Error during knn_query: {e}")
            indices_henn, distances_henn = np.array([]), np.array([])

        found_labels = [labels[i] for i in indices_henn[0]] if len(indices_henn) > 0 else []
        
        # Write HENN results to file
        with open('henn_knn_results.txt', 'a') as f:
            f.write(f"Query: {query.tolist()}\n")
            f.write(f"Labels: {found_labels}\n")
            f.write(f"K: {i+3}\n")
            f.write(f"Distances: {distances_henn.tolist()}\n")
            f.write("-" * 50 + "\n")
        f.close()


if __name__ == "__main__":
    # test_basic_functionality()
    test_range_use_random()