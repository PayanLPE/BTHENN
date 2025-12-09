#pragma once

#include "../hnswlib/henn.h"
#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>
#include <string>
#include <queue> // Added for priority_queue

using namespace std;
using namespace hnswlib;
using namespace henn;

namespace btree_henn
{

    // class for the node present in a B-Tree
    template <typename T, int ORDER>
    class BTreeNode
    {
    public:
        // Array of keys
        pair<T, int> keys[ORDER - 1];
        // Array of child pointers
        BTreeNode *children[ORDER];
        // Current number of keys
        int n;
        // True if leaf node, false otherwise
        bool leaf;
        // Flag to identify dummy/temporary nodes for cleanup
        bool is_temporary = false; 

        // Keep track of all henn data in subtree
        vector<int> henn_indices;
        float *henn_data;

        // store HENN index for this node
        unique_ptr<HierarchicalNSW<float>> henn_index;

        BTreeNode(bool isLeaf = true) : n(0), leaf(isLeaf), henn_data(nullptr)
        {
            for (int i = 0; i < ORDER; i++)
                children[i] = nullptr;
        }

        ~BTreeNode() {
            if (henn_data) {
                delete[] henn_data;
            }
            // unique_ptr handles henn_index deletion automatically
        }

        auto searchKNN(const void *query_data, int k)
        {
            if (henn_index && !henn_indices.empty())
            {
                auto results = henn_index->searchKnn(query_data, k);
                // Map the labels back to original indices
                std::priority_queue<std::pair<float, hnswlib::labeltype>> mapped_results;

                while (!results.empty())
                {
                    auto pair = results.top();
                    results.pop();
                    // pair.second is the label in the HENN index (0-based)
                    // Map it back to the original index using henn_indices
                    if (pair.second < henn_indices.size())
                    {
                        hnswlib::labeltype original_index = henn_indices[pair.second];
                        mapped_results.push({pair.first, original_index});
                    }
                }
                return mapped_results;
            }
            else
            {
                // No index, return empty result
                return std::priority_queue<std::pair<float, hnswlib::labeltype>>();
            }
        }
    };

    // class for B-Tree
    template <typename T, int ORDER>
    class BTree
    {
    private:
        BTreeNode<T, ORDER> *root;               // Pointer to root node
        float *henn_data;                        // HENN data for all elements
        int dim;                                 // Dimension of data points
        int n;                                   // Number of elements
        hnswlib::SpaceInterface<float> *l2space; // L2 space for HENN

        // Function to split a full child node
        void splitChild(BTreeNode<T, ORDER> *x, int i)
        {
            BTreeNode<T, ORDER> *y = x->children[i];
            BTreeNode<T, ORDER> *z = new BTreeNode<T, ORDER>(y->leaf);

            int mid = ORDER / 2;
            z->n = ORDER - 1 - mid;

            // Copy the last half of keys to new node
            for (int j = 0; j < z->n; j++)
                z->keys[j] = y->keys[j + mid];

            if (!y->leaf)
            {
                // Copy children pointers
                for (int j = 0; j <= z->n; j++)
                    z->children[j] = y->children[j + mid];
            }

            y->n = mid - 1;

            // Make room for new child in parent
            for (int j = x->n; j >= i + 1; j--)
                x->children[j + 1] = x->children[j];

            x->children[i + 1] = z;

            // Make room for promoted key in parent
            for (int j = x->n - 1; j >= i; j--)
                x->keys[j + 1] = x->keys[j];

            // Promote the middle key
            x->keys[i] = y->keys[mid - 1];
            x->n = x->n + 1;
        }

        // Function to insert a key in a non-full node
        void insertNonFull(BTreeNode<T, ORDER> *x, T k, int index)
        {
            int i = x->n - 1;

            if (x->leaf)
            {
                while (i >= 0 && k < x->keys[i].first)
                {
                    x->keys[i + 1] = x->keys[i];
                    i--;
                }

                x->keys[i + 1] = make_pair(k, index);
                x->n = x->n + 1;
            }
            else
            {
                while (i >= 0 && k < x->keys[i].first)
                    i--;

                i++;
                if (x->children[i]->n == ORDER - 1)
                {
                    splitChild(x, i);

                    if (k > x->keys[i].first)
                        i++;
                }
                insertNonFull(x->children[i], k, index);
            }
        }

        // Helper to recursively build HENN
        void build_henn_recursive(BTreeNode<T, ORDER> *node)
        {
            if (node == nullptr)
                return;

            // If leaf, collect keys' data
            for (int i = 0; i < node->n; i++)
            {
                node->henn_indices.push_back(node->keys[i].second);
            }
            if (!node->leaf)
            {
                // For non-leaf nodes, recursively build HENN indexes for children
                for (int i = 0; i <= node->n; i++)
                {
                    build_henn_recursive(node->children[i]);
                }

                // Collect indices from all children
                for (int i = 0; i <= node->n; i++)
                {
                    if (node->children[i])
                    {
                        node->henn_indices.insert(
                            node->henn_indices.end(),
                            node->children[i]->henn_indices.begin(),
                            node->children[i]->henn_indices.end());
                    }
                }
            }

            if (node->henn_indices.empty())
                return;

            // Allocate memory for the node's HENN data
            node->henn_data = new float[dim * node->henn_indices.size()];

            // Copy data for each index in this node
            for (size_t j = 0; j < node->henn_indices.size(); j++)
            {
                int idx = node->henn_indices[j];
                // Copy the feature vector for this index
                for (int d = 0; d < dim; d++)
                {
                    node->henn_data[j * dim + d] = henn_data[idx * dim + d];
                }
            }

            int M = 16;
            node->henn_index = unique_ptr<HierarchicalNSW<float>>(
                henn::buildHENN(node->henn_data, node->henn_indices.size(), dim, l2space, M, true));
        }

        // Helper to create dummy node
        BTreeNode<T, ORDER> *createDummyNodeFromIndices(const vector<pair<T, int>> &indices_not_covered)
        {
            if (indices_not_covered.empty() || !l2space || !henn_data)
                return nullptr;

            BTreeNode<T, ORDER> *dummy_node = new BTreeNode<T, ORDER>(true);
            dummy_node->is_temporary = true; // Mark for deletion
            
            size_t m = indices_not_covered.size();
            
            // copy keys
            for (size_t i = 0; i < m && i < ORDER - 1; ++i)
                dummy_node->keys[i] = indices_not_covered[i];
            
            dummy_node->n = (m < ORDER - 1) ? static_cast<int>(m) : ORDER - 1;

            dummy_node->henn_indices.reserve(m);
            for (size_t i = 0; i < m; ++i)
                dummy_node->henn_indices.push_back(indices_not_covered[i].second);

            // copy data
            dummy_node->henn_data = new float[dim * dummy_node->henn_indices.size()];
            for (size_t j = 0; j < dummy_node->henn_indices.size(); ++j)
            {
                int idx = dummy_node->henn_indices[j];
                if (idx >= 0 && idx < n) {
                     for (int d = 0; d < dim; ++d)
                        dummy_node->henn_data[j * dim + d] = henn_data[idx * dim + d];
                }
            }

            int M = 16; 
            dummy_node->henn_index = unique_ptr<HierarchicalNSW<float>>(
                henn::buildHENN(dummy_node->henn_data, dummy_node->henn_indices.size(), dim, l2space, M, true));

            return dummy_node;
        }

        void printTreeStructureHelper(BTreeNode<T, ORDER> *node, int level, const string &prefix) const
        {
            if (node == nullptr) return;
            for (int i = 0; i < level; i++) cout << "    ";
            cout << prefix;
            if (node->leaf) cout << "[LEAF] ";
            else cout << "[INTERNAL] ";
            cout << "Keys(" << node->n << "): [";
            for (int i = 0; i < node->n; i++)
            {
                cout << node->keys[i].first;
                if (i < node->n - 1) cout << ", ";
            }
            cout << "]" << endl;
            if (!node->leaf)
            {
                for (int i = 0; i <= node->n; i++)
                {
                    printTreeStructureHelper(node->children[i], level + 1, "Child[" + to_string(i) + "]: ");
                }
            }
        }

    public:
        // FIX 1: Initialize pointers to nullptr
        BTree() : root(new BTreeNode<T, ORDER>(true)), henn_data(nullptr), l2space(nullptr), dim(0), n(0) {}

        ~BTree()
        {
            if (l2space) delete l2space;
        }

        // Function to insert a key in the tree
        void insert(T k, int index)
        {
            if (root->n == ORDER - 1)
            {
                BTreeNode<T, ORDER> *s = new BTreeNode<T, ORDER>(false);
                s->children[0] = root;
                root = s;
                splitChild(s, 0);
                insertNonFull(s, k, index);
            }
            else
                insertNonFull(root, k, index);
        }

        void build_henn_index(float *data, int num_elements, int dimension)
        {
            henn_data = data;
            if (l2space) delete l2space;
            l2space = new hnswlib::L2Space(dimension);
            dim = dimension;
            n = num_elements;
            build_henn_recursive(root);
        }

        vector<BTreeNode<T, ORDER> *> findCanonicalNodes(T k1, T k2)
        {
            vector<BTreeNode<T, ORDER> *> canonical_nodes;
            BTreeNode<T, ORDER> *node = root;
            BTreeNode<T, ORDER> *split_node = nullptr;
            int i1 = 0, i2 = 0; // Declare here to be accessible after the loop
            
            if (node == nullptr || l2space == nullptr)
                return canonical_nodes;

            // Find the split node
            while (!node->leaf)
            {
                // Route k1
                i1 = 0;
                while (i1 < node->n && k1 >= node->keys[i1].first) i1++;

                // Route k2
                i2 = 0;
                while (i2 < node->n && k2 >= node->keys[i2].first) i2++;

                // In the same range, go to that child
                if (i1 == i2)
                {
                    node = node->children[i1];
                }
                else
                {
                    split_node = node;
                    // Add the children of the split node between i1 and i2
                    for (int j = i1 + 1; j < i2; j++)
                    {
                        canonical_nodes.push_back(split_node->children[j]);
                    }
                    break;
                }
            }

            vector<pair<T, int>> indices_not_covered;

            if (split_node == nullptr)
            {
                i1 = 0;
                while (i1 < node->n && k1 > node->keys[i1].first) i1++; 
                for(int i=0; i<node->n; i++) {
                    if (node->keys[i].first >= k1 && node->keys[i].first <= k2) {
                        indices_not_covered.push_back(node->keys[i]);
                    }
                }
            }
            else
            {
                for (int i = i1; i < i2; i++)
                {
                    indices_not_covered.push_back(split_node->keys[i]);
                }

                BTreeNode<T, ORDER> *left_node = split_node->children[i1];
                while (left_node != nullptr)
                {
                    int index = 0;
                    bool found_split = false;

                    for (int i = 0; i < left_node->n; i++)
                    {
                        if (k1 <= left_node->keys[i].first)
                        {
                            // Add right-side siblings
                            for (int j = i + 1; j <= left_node->n; j++)
                                if (!left_node->leaf && left_node->children[j]) 
                                    canonical_nodes.push_back(left_node->children[j]);

                            // Add keys >= k1
                            for (int k = i; k < left_node->n; k++)
                                indices_not_covered.push_back(left_node->keys[k]);

                            index = i;
                            found_split = true;
                            break;
                        }
                    }

                    if (!found_split) index = left_node->n;
                    
                    if (left_node->leaf) break; // Terminate loop if we are at leaf
                    left_node = left_node->children[index];
                }

                // Right path: from split_node->children[i2] down to leaf
                BTreeNode<T, ORDER> *right_node = split_node->children[i2];
                while (right_node != nullptr)
                {
                    int index = 0; 
                    bool found_split = false;
                    for (int i = right_node->n - 1; i >= 0; i--)
                    {
                        if (k2 >= right_node->keys[i].first)
                        {
                            // Add left-side siblings
                            for (int j = 0; j <= i; j++)
                                if (!right_node->leaf && right_node->children[j]) 
                                    canonical_nodes.push_back(right_node->children[j]);

                            // Add keys <= k2
                            for (int k = 0; k <= i; k++)
                                indices_not_covered.push_back(right_node->keys[k]);

                            index = i + 1;
                            found_split = true;
                            break;
                        }
                    }
                    if (!found_split) index = 0; // If k2 is smaller than all, go leftmost

                    if (right_node->leaf) break; // Terminate loop if we are at leaf
                    right_node = right_node->children[index];
                }
            }

            if (!indices_not_covered.empty())
            {
                BTreeNode<T, ORDER> *dummy_node = createDummyNodeFromIndices(indices_not_covered);
                if(dummy_node) canonical_nodes.push_back(dummy_node);
            }

            return canonical_nodes;
        }

        auto rangeSearchKNN(T k1, T k2, float *data, int k)
        {
            auto result = std::priority_queue<std::pair<float, hnswlib::labeltype>>();
            auto nodes = findCanonicalNodes(k1, k2);
            for (auto node : nodes)
            {
                auto node_results = node->searchKNN(data, k);
                while (!node_results.empty())
                {
                    result.push(node_results.top());
                    node_results.pop();
                }
                // Keep only top k global results (Optimization)
                while (result.size() > k)
                {
                    result.pop();
                }

                // Cleanup dummy node
                if (node->is_temporary) {
                    delete node;
                }
            }
            return result;
        }

        void printTreeStructure() const
        {
            if (root) printTreeStructureHelper(root, 0, "Root: ");
        }
    };
}