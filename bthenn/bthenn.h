#pragma once

#include "../hnswlib/henn.h"
#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>
#include <string>

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

        // Keep track of all henn data in subtree
        vector<int> henn_indices;
        float *henn_data;

        // store HENN index for this node
        unique_ptr<HierarchicalNSW<float>> henn_index;

        BTreeNode(bool isLeaf = true) : n(0), leaf(isLeaf)
        {
            for (int i = 0; i < ORDER; i++)
                children[i] = nullptr;
        }

        auto searchKNN(const void *query_data, int k)
        {
            if (henn_index)
            {
                return henn_index->searchKnn(query_data, k);
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

        // Function to traverse the tree
        void traverse(BTreeNode<T, ORDER> *x)
        {
            int i;
            for (i = 0; i < x->n; i++)
            {
                if (!x->leaf)
                    traverse(x->children[i]);
                cout << " " << x->keys[i].first;
            }

            if (!x->leaf)
                traverse(x->children[i]);
        }

    public:
        BTree(int dimension = 128) : dim(dimension)
        {
            root = new BTreeNode<T, ORDER>(true);
            l2space = new hnswlib::L2Space(dim);
        }

        ~BTree()
        {
            delete l2space;
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
            dim = dimension;
            n = num_elements;
            build_henn_recursive(root);
        }

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

            // Create HENN index for this node if it has enough points
            if (node->henn_indices.size() > 1)
            { // Need at least 2 points for HENN
                // Use HENN algorithm to build hierarchical index
                int M = 1;        // Hierarchy parameter - you can tune this
                bool best = true; // Use best epsilon nets

                node->henn_index = unique_ptr<HierarchicalNSW<float>>(
                    henn::buildHENN(node->henn_data, node->henn_indices.size(), dim, l2space, M, best));
            }
            else if (node->henn_indices.size() == 1)
            {
                // For single point, create simple HNSW index
                node->henn_index = make_unique<HierarchicalNSW<float>>(l2space, 1, 16, 200);
                node->henn_index->addPoint(&node->henn_data[0], 0);
            }
        }

        // Function to traverse the tree
        void traverse()
        {
            if (root != nullptr)
                traverse(root);
        }

        // Function to range query in the tree
        std::priority_queue<std::pair<T, hnswlib::labeltype>> rangeSearchKNN(T k1, T k2, int k)
        {
            if (root == nullptr)
            {
                return std::priority_queue<std::pair<T, hnswlib::labeltype>>();
                
            }

            std::priority_queue<std::pair<T, hnswlib::labeltype>> result = rangeSearchKNN(root, k1, k2, k);
            return result;
        }

        // Function to find canonical nodes for range [k1, k2]
        vector<BTreeNode<T, ORDER> *> findCanonicalNodes(T k1, T k2)
        {
            vector<BTreeNode<T, ORDER> *> canonical_nodes;
            BTreeNode<T, ORDER> *node = root;
            BTreeNode<T, ORDER> *split_node = nullptr;
            int i1 = 0, i2 = 0; // Declare here to be accessible after the loop
            
            if (node == nullptr)
                return canonical_nodes;

            // Find the split node
            while (!node->leaf)
            {
                // Route k1
                i1 = 0;
                while (i1 < node->n && k1 >= node->keys[i1].first)
                    i1++;
                
                // Route k2
                i2 = 0;
                while (i2 < node->n && k2 >= node->keys[i2].first)
                    i2++;
                
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
            cout << "Split node keys: ";
            for (auto node : split_node->keys)
            {
                cout << node.first << " ";
            }
            cout << endl;

            
            if (split_node == nullptr)
                return canonical_nodes;

            // Left path: walk down to leaf, add all the children nodes to the right
            BTreeNode<T, ORDER> *left_node = split_node->children[i1];
            while (!left_node->leaf)
            {
                int index = 0;
                for (int i = 0; i < left_node->n; i++)
                {
                    if (k1 <= left_node->keys[i].first)
                    {
                        // Add all children to the right of the current position
                        for (int j = i + 1; j <= left_node->n; j++)
                        {
                            canonical_nodes.push_back(left_node->children[j]);
                        }
                        break;
                    }
                    else
                    {
                        index = i + 1;
                    }
                }
                left_node = left_node->children[index];
            }

            // Right path: walk down to leaf, add all the children nodes to the left
            BTreeNode<T, ORDER> *right_node = split_node->children[i2];
            while (!right_node->leaf)
            {
                int index = right_node->n; // Default to rightmost child
                for (int i = right_node->n - 1; i >= 0; i--)
                {
                    if (k2 >= right_node->keys[i].first)
                    {
                        // Add all children to the left of the current position
                        for (int j = 0; j <= i; j++)
                        {
                            canonical_nodes.push_back(right_node->children[j]);
                        }
                        index = i + 1;
                        break;
                    }
                }
                right_node = right_node->children[index];
            }

            return canonical_nodes;
        }

        // Function to search a key in the tree
        std::priority_queue<std::pair<T, hnswlib::labeltype>> rangeSearchKNN(T k1, T k2, int k)
        {
            auto result = priority_queue<std::pair<T, hnswlib::labeltype>>();
            auto nodes  = findCanonicalNodes(k1, k2);
            for (auto node : nodes) {
                cout << "Canonical node keys: ";
                for (int i = 0; i < node->n; i++) {
                    cout << node->keys[i].first << " ";
                }
                cout << endl;
               auto node_results = node->henn_index->searchKnn(node->henn_data.data(), k);
               for (; !node_results.empty(); node_results.pop()) {
                   result.push(node_results.top());
               }
            }
            return result;
        }

        // Function to get the root node (for testing and debugging)
        BTreeNode<T, ORDER> *getRoot() const
        {
            return root;
        }

        // Function to print detailed tree structure
        void printTreeStructure() const
        {
            cout << "=== B-Tree Structure (ORDER=" << ORDER << ") ===" << endl;
            if (root == nullptr)
            {
                cout << "Tree is empty" << endl;
                return;
            }
            printTreeStructureHelper(root, 0, "Root: ");
            cout << endl;
        }

    private:
        // Helper function to print tree structure recursively
        void printTreeStructureHelper(BTreeNode<T, ORDER> *node, int level, const string &prefix) const
        {
            if (node == nullptr)
                return;

            // Print indentation for current level
            for (int i = 0; i < level; i++)
            {
                cout << "    ";
            }

            cout << prefix;

            // Print node type
            if (node->leaf)
            {
                cout << "[LEAF] ";
            }
            else
            {
                cout << "[INTERNAL] ";
            }

            // Print keys in this node
            cout << "Keys(" << node->n << "): [";
            for (int i = 0; i < node->n; i++)
            {
                cout << node->keys[i].first << " (Index: " << node->keys[i].second << ")";
                if (i < node->n - 1)
                    cout << ", ";
            }
            cout << "]";
            cout << "; HENN Indices: {";
            for (size_t i = 0; i < node->henn_indices.size();
                 i++)
            {
                cout << node->henn_indices[i];
                if (i < node->henn_indices.size() - 1)
                    cout << ", ";
            }
            cout << "}";

            // Print HENN data information
            if (node->henn_data)
            {
                cout << "; HENN Data: [";
                for (size_t i = 0; i < dim * node->henn_indices.size(); i++)
                {
                    cout << node->henn_data[i];
                    if (i < dim * node->henn_indices.size() - 1)
                        cout << ", ";
                }
                cout << "]";
            }
            else
            {
                cout << "; No HENN Data";
            }

            // Print memory address for debugging
            cout << " @" << node << endl;

            // Recursively print children
            if (!node->leaf)
            {
                for (int i = 0; i <= node->n; i++)
                {
                    string childPrefix = "Child[" + to_string(i) + "]: ";
                    printTreeStructureHelper(node->children[i], level + 1, childPrefix);
                }
            }
        }
    };
}