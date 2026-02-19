#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include <random>
#include <map>

#include "hnswlib/hnswlib.h"
#include "hnswlib/endtree.h"

int main() {
    std::cout << "\n=== EndTree Structure Demo with Random Data ===\n\n";

    // Generate random 2D data
    const int dim = 2;
    const int n = 50;  // More points for better hierarchy
    
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);
    
    std::vector<float> data(static_cast<size_t>(n * dim));
    std::cout << "Generated " << n << " random points in " << dim << "D space\n";
    // std::cout << "(showing first 10):\n\n";
    
    for (int i = 0; i < n; ++i) {
        data[i * dim] = dist(rng);
        data[i * dim + 1] = dist(rng);
        std::cout << "  Point " << std::setw(2) << i << ": (" 
                      << std::fixed << std::setprecision(2) << std::setw(6) << data[i * dim] 
                      << ", " << std::setw(6) << data[i * dim + 1] << ")\n";
    }
    std::cout << "  ...\n";

    // Build HENN structure manually with explicit layers for demo
    hnswlib::L2Space space(dim);
    std::cout << "\n--- Building EndTree (HENN + Augmentation) ---\n";
    
    // Create index and manually assign layers
    auto index = std::unique_ptr<hnswlib::HierarchicalNSW<float>>(
        new hnswlib::HierarchicalNSW<float>(&space, n, 16));
    std::vector<int> max_levels(n);
    
    // Assign layers: ~10% layer 2, ~30% layer 1, rest layer 0
    for (int i = 0; i < n; ++i) {
        if (i < n / 10) {
            max_levels[i] = 2;  // Top layer
            index->addPoint(&data[i * dim], i, 2);
        } else if (i < 4 * n / 10) {
            max_levels[i] = 1;  // Middle layer
            index->addPoint(&data[i * dim], i, 1);
        } else {
            max_levels[i] = 0;  // Base layer
            index->addPoint(&data[i * dim], i, 0);
        }
    }
    index->setEf(100);
    
    // Augment with parent-child statistics
    auto aug = endtree::augment_bottom_up<float>(
        index.get(), data.data(), n, dim, &space, max_levels, 100, 200);
    
    // Package into EndTree structure
    endtree::EndTreeIndex<float> tree;
    tree.index = std::move(index);
    tree.max_levels = max_levels;
    tree.augmentation = aug;
    
    std::cout << "\nHierarchical layer distribution:\n";
    std::map<int, std::vector<int>> layer_map;
    int max_level = 0;
    for (int i = 0; i < n; ++i) {
        int level = tree.max_levels[i];
        layer_map[level].push_back(i);
        max_level = std::max(max_level, level);
    }
    
    for (int layer = max_level; layer >= 0; --layer) {
        std::cout << "  Layer " << layer << ": ";
        if (layer_map[layer].size() <= 15) {
            std::cout << "[";
            for (size_t i = 0; i < layer_map[layer].size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << layer_map[layer][i];
            }
            std::cout << "]";
        } else {
            std::cout << "[";
            for (size_t i = 0; i < 10; ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << layer_map[layer][i];
            }
            std::cout << ", ...]";
        }
        std::cout << "  (" << layer_map[layer].size() << " points)\n";
    }

    std::cout << "\n--- Parent-Child Augmentation ---\n";
    std::cout << "\nSample parent-child relationships:\n";
    
    int shown = 0;
    for (int i = 0; i < n && shown < 15; ++i) {
        if (tree.augmentation.parent_of[i] != endtree::invalid_label) {
            auto parent = tree.augmentation.parent_of[i];
            float dx = data[i * dim] - data[parent * dim];
            float dy = data[i * dim + 1] - data[parent * dim + 1];
            float dist_sq = dx * dx + dy * dy;
            
            std::cout << "  Point " << std::setw(2) << i << " (L" << tree.max_levels[i] 
                      << ") → Parent " << std::setw(2) << parent << " (L" << tree.max_levels[parent] 
                      << "), dist² = " << std::fixed << std::setprecision(1) << std::setw(8) << dist_sq << "\n";
            shown++;
        }
    }
    if (shown == 0) {
        std::cout << "  (No parent-child relationships - all points in top layer)\n";
    }

    std::cout << "\nSample parent statistics:\n";
    shown = 0;
    for (const auto& parent_entry : tree.augmentation.parent_stats) {
        if (shown >= 8) break;
        
        int parent_label = parent_entry.first;
        std::cout << "  Parent " << std::setw(2) << parent_label << " (L" << tree.max_levels[parent_label] << "): ";
        
        size_t total_children = 0;
        for (const auto& layer_entry : parent_entry.second) {
            total_children += layer_entry.second.count;
        }
        std::cout << total_children << " children, ";
        
        for (const auto& layer_entry : parent_entry.second) {
            const auto& stats = layer_entry.second;
            std::cout << "radius=" << std::fixed << std::setprecision(1) << stats.coverage_radius;
            break;  // Just show first layer stats
        }
        std::cout << "\n";
        shown++;
    }

    std::cout << "\n--- Outlier Score Testing ---\n";
    
    // Test various query points
    std::vector<std::pair<float, float>> queries = {
        {50.0f, 50.0f},     // Center
        {0.0f, 0.0f},       // Corner
        {100.0f, 100.0f},   // Opposite corner
        {data[0], data[1]}, // On an existing point
        {150.0f, 150.0f}    // Far outlier
    };
    
    std::cout << "\nTesting " << queries.size() << " query points:\n";
    
    for (size_t q_idx = 0; q_idx < queries.size(); ++q_idx) {
        float q[2] = {queries[q_idx].first, queries[q_idx].second};
        auto score = endtree::query_outlier_score_path(tree, q);
        
        std::cout << "\n  Query " << (q_idx + 1) << ": (" 
                  << std::fixed << std::setprecision(1) << std::setw(6) << q[0] 
                  << ", " << std::setw(6) << q[1] << ")";
        
        // Show search path
        std::cout << "  →  ";
        for (int layer = static_cast<int>(score.layers.size()) - 1; layer >= 0; --layer) {
            if (layer < static_cast<int>(score.layers.size()) - 1) std::cout << "→";
            std::cout << "L" << layer << "[" << std::setw(2) << score.layers[layer].p << "]";
        }
        
        // Show anomaly score
        std::cout << "  Score: ";
        if (std::isfinite(score.total_score) && score.total_score < 1e6) {
            std::cout << std::fixed << std::setprecision(4) << score.total_score;
        } else if (score.total_score == 0.0) {
            std::cout << "0.0000 (exact match)";
        } else {
            std::cout << std::fixed << std::setprecision(4) << score.total_score;
        }
        std::cout << "\n";
    }

    std::cout << "\n=== Summary ===\n";
    std::cout << "\nEndTree Structure:\n";
    std::cout << "  • " << n << " points organized in " << (max_level + 1) << " layers\n";
    std::cout << "  • " << tree.augmentation.parent_stats.size() << " parents with child statistics\n";
    std::cout << "  • Each parent tracks: count, coverage radius, mean, variance\n";
    std::cout << "  • Anomaly scores combine distance, density, and structural deviation\n\n";

    return 0;
}
