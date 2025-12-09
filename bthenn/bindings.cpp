#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "bthenn.h"

namespace py = pybind11;
using namespace pybind11::literals;

template <typename T, int ORDER>
class BtHennIndex
{
public:
    static const int ser_version = 1; // serialization version

    btree_henn::BTree<T, ORDER> *btree;

    BtHennIndex()
    {
        btree = new btree_henn::BTree<T, ORDER>();
    }

    ~BtHennIndex()
    {
        // delete btree;
    }

    void buildBtreeHENN(py::array_t<float> data, py::array_t<float> label, int num_elements, int dimension)
    {
        py::buffer_info buf = data.request();
        if (buf.shape[0] != num_elements || buf.shape[1] != dimension || buf.shape[0] != label.request().shape[0])
        {
            throw std::runtime_error("Array dimensions don't match specified parameters");
        }

        // Insert all the labels into the B-Tree
        for (size_t i = 0; i < label.request().shape[0]; i++)
        {
            T key = *(static_cast<T *>(label.request().ptr) + i);
            btree->insert(key, i);
        }
        btree->build_henn_index(static_cast<float *>(buf.ptr), num_elements, dimension);
    }

    void insert(T key, int index)
    {
        btree->insert(key, index);
    }

    void findCanonicalNodes(T min, T max)
    {
        auto nodes = btree->findCanonicalNodes(min, max);
        for (auto node : nodes)
        {
            std::cout << "Node with keys: ";
            for (int j = 0; j < node->n; j++)
            {
                std::cout << node->keys[j].first << " ";
            }
            std::cout << std::endl;
        }
    }

    void printTreeStructure()
    {
        btree->printTreeStructure();
    }

    std::vector<std::pair<float, size_t>> rangeSearchKNN(T k1, T k2, py::array_t<float> query, int k)
    {
        py::buffer_info buf = query.request();
        if (buf.ndim != 1)
        {
            throw std::runtime_error("Query must be 1-dimensional");
        }
        auto pq = btree->rangeSearchKNN(k1, k2, static_cast<float *>(buf.ptr), k);

        // Convert priority queue to vector
        std::vector<std::pair<float, size_t>> result;
        while (!pq.empty())
        {
            result.push_back(pq.top());
            pq.pop();
        }
        std::reverse(result.begin(), result.end()); // Reverse to get closest first
        return result;
    }
};

PYBIND11_MODULE(bthenn, m)
{
    py::class_<BtHennIndex<float, 16>>(m, "Index16")
        .def(py::init<>())
        .def("build_btree_henn", &BtHennIndex<float, 16>::buildBtreeHENN, "data"_a, "label"_a, "num_elements"_a, "dimension"_a,
             "Build B-Tree with HENN indexes")
        .def("insert", &BtHennIndex<float, 16>::insert, "key"_a, "index"_a,
             "Insert a key-index pair into the B-Tree")
        .def("findCanonicalNodes", &BtHennIndex<float, 16>::findCanonicalNodes, "min"_a, "max"_a,
             "Find canonical nodes for range [min, max]")
        .def("print_tree_structure", &BtHennIndex<float, 16>::printTreeStructure,
             "Print detailed tree structure for debugging")
        .def("rangeSearchKNN", &BtHennIndex<float, 16>::rangeSearchKNN, "k1"_a, "k2"_a, "query"_a, "k"_a,
             "Perform range KNN search for keys in [k1, k2]");

    py::class_<BtHennIndex<float, 32>>(m, "Index32")
        .def("build_btree_henn", &BtHennIndex<float, 32>::buildBtreeHENN, "data"_a, "label"_a, "num_elements"_a, "dimension"_a,
             "Build B-Tree with HENN indexes");

    py::class_<BtHennIndex<float, 64>>(m, "Index64")
        .def("build_btree_henn", &BtHennIndex<float, 64>::buildBtreeHENN, "data"_a, "label"_a, "num_elements"_a, "dimension"_a,
             "Build B-Tree with HENN indexes");
}
