# 🌳 BT-HENN & EndTree (Extended HENN)

*Hierarchical ε-Net Navigation Graphs for Hybrid Range Queries and Anomaly Detection*

<p align="center">
  <img alt="HENN Structure" src="examples/reports/nav_graph_darkbg.png" width="70%">
</p>

## Overview

This repository builds on [hnswlib](https://github.com/nmslib/hnswlib) by introducing the **HENN** (Hierarchical $\epsilon$-Net Navigation) graph and extending it to solve two major limitations in modern Approximate Nearest Neighbor (ANN) search: empty-set failures in post-filtering, and the immense overhead of exact outlier detection.

This fork introduces three core architectures:
1. **HENN**: The base hierarchical structure using deterministic $\epsilon$-nets to guarantee polylogarithmic ANN search.
2. **BT-HENN (B-Tree HENN)**: Integrates HENN sub-indices within a balanced B-tree over a numerical attribute. This structurally guarantees adherence to strict range constraints $(a, b)$ with a perfect recall of 1.0, bypassing the failure states of naive post-filtering. 
3. **EndTree**: Embeds local density and distance statistics directly into the $\epsilon$-net layers during index construction. It calculates Local Outlier Factor (LOF)-style anomaly scores concurrently during standard ANN traversals, eliminating the need for expensive $\mathcal{O}(n^2)$ secondary graph constructions.

## Repository Structure

- Forked from [hnswlib](https://github.com/nmslib/hnswlib)
- **Core implementations:** - Base HENN: [`hnswlib/henn.h`](hnswlib/henn.h)
  - Range-Aware BT-HENN: [`hnswlib/bt_henn.h`](hnswlib/bt_henn.h)
  - Outlier Detection (EndTree): [`hnswlib/endtree.h`](hnswlib/endtree.h)
- **Example usage:**
  - C++: `examples/cpp/henn`, `examples/cpp/bt_henn`, `examples/cpp/endtree`
  - Python: `examples/python/henn`, `examples/python/bt_henn`, `examples/python/endtree`

*Note: To run the C++ examples, edit `CMakeLists.txt` to compile your desired targets.*

## Build and Run

Follow the original [hnswlib build instructions](https://github.com/nmslib/hnswlib):

### C++ Build

```bash
mkdir build && cd build
cmake ..
make

Extending Hierarchical Epsilon-Net Navigation (HENN) Graphs for Hybrid Range Queries and Hierarchical Outlier Detection (Liu, 2026).
