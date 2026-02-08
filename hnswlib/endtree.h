#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "henn.h"

// EndTree: HENN + a bottom-up augmentation pass.
//
// After constructing the HENN hierarchy (same as `henn::buildHENN`), we assign each point
// with max level i to a "parent" in the layer above (any point with max level >= i+1).
// For every parent, we aggregate statistics over its children from layer i:
//   - count (density)
//   - coverage radius (max distance parent->child)
//   - mean/variance of parent->child distances
//
// Notes:
// - Distances are in the units returned by `space->get_dist_func()` / HNSW (e.g. L2 may be squared).
// - Parent selection uses HNSW search with a filter, with an exact-scan fallback if needed.

namespace endtree {

using hnswlib::labeltype;

static constexpr labeltype invalid_label = std::numeric_limits<labeltype>::max();

struct ChildStats {
	size_t count = 0;
	float coverage_radius = 0.0f;
	double mean = 0.0;
	double variance = 0.0;

	// Internal for Welford's algorithm
	double m2 = 0.0;

	void add(double dist) {
		count += 1;
		if (dist > static_cast<double>(coverage_radius)) {
			coverage_radius = static_cast<float>(dist);
		}

		// Welford online mean/variance
		double delta = dist - mean;
		mean += delta / static_cast<double>(count);
		double delta2 = dist - mean;
		m2 += delta * delta2;
		variance = (count > 1) ? (m2 / static_cast<double>(count - 1)) : 0.0;
	}
};

// stats[parent][child_layer] -> aggregated stats of children whose max layer == child_layer
using ParentStatsByLayer = std::unordered_map<int, ChildStats>;

struct Augmentation {
	// parent_of[u] is the chosen parent for u (only defined for nodes that are not in the top layer).
	// If no parent exists (e.g. u is in top layer), value is `invalid_label`.
	std::vector<labeltype> parent_of;

	// Aggregated statistics stored on parents.
	std::unordered_map<labeltype, ParentStatsByLayer> parent_stats;
};

template <typename dist_t>
struct EndTreeIndex {
	std::unique_ptr<hnswlib::HierarchicalNSW<dist_t>> index;
	std::vector<int> max_levels;
	Augmentation augmentation;
};

struct LayerOutlierScore {
	int layer = -1;
	labeltype p = invalid_label;
	float dist_q_p = std::numeric_limits<float>::infinity();

	// Scores for this layer (only defined when there is a parent in layer+1).
	bool has_parent_stats = false;
	float parent_coverage_radius = 0.0f;
	size_t parent_count = 0;
	double parent_mean = 0.0;
	double parent_variance = 0.0;

	double score_dist = 0.0;
	double score_density = 0.0;
	double score_struct = 0.0;
	double s = 0.0;

	double weight = 1.0;
	double weighted_s = 0.0;
};

struct OutlierScoreResult {
	// layers[0] is level 0, layers[max_level] is top.
	std::vector<LayerOutlierScore> layers;
	double total_score = 0.0;
};

namespace detail {

static constexpr labeltype invalid_label = endtree::invalid_label;

// Filter for HNSW search that only allows nodes with max_level >= parent_min_level as parents.
class MinLevelFilter final : public hnswlib::BaseFilterFunctor {
  public:
	MinLevelFilter(const std::vector<int>* max_levels, int min_level)
		: max_levels_(max_levels), min_level_(min_level) {}

	bool operator()(hnswlib::labeltype id) override {
		if (max_levels_ == nullptr) return false;
		if (id >= max_levels_->size()) return false;
		return (*max_levels_)[id] >= min_level_;
	}

  private:
	const std::vector<int>* max_levels_;
	int min_level_;
};

inline ChildStats get_child_stats_or_default(
	const std::unordered_map<labeltype, ParentStatsByLayer>& parent_stats,
	labeltype parent,
	int child_layer,
	bool* found) {

	if (found) *found = false;
	auto it_parent = parent_stats.find(parent);
	if (it_parent == parent_stats.end()) {
		return ChildStats{};
	}
	auto it_layer = it_parent->second.find(child_layer);
	if (it_layer == it_parent->second.end()) {
		return ChildStats{};
	}
	if (found) *found = true;
	return it_layer->second;
}

template <typename dist_t>
static std::pair<labeltype, dist_t> exact_nearest_parent(
	const float* points,
	size_t numPoints,
	size_t dim,
	hnswlib::SpaceInterface<dist_t>* space,
	const std::vector<int>& max_levels,
	labeltype u,
	int parent_min_level) {

	const auto dist_func = space->get_dist_func();
	const auto dist_param = space->get_dist_func_param();

	labeltype best_parent = invalid_label;
	dist_t best_dist = std::numeric_limits<dist_t>::max();

	const float* u_ptr = points + static_cast<size_t>(u) * dim;

	for (labeltype v = 0; v < numPoints; ++v) {
		if (max_levels[v] < parent_min_level) continue;
		// v is a valid parent candidate
		dist_t d = dist_func((const void*)u_ptr, (const void*)(points + static_cast<size_t>(v) * dim), dist_param);
		if (d < best_dist) {
			best_dist = d;
			best_parent = v;
		}
	}

	return {best_parent, best_dist};
}

}  // namespace detail

namespace detail {

inline std::pair<hnswlib::tableint, float> greedy_closest_in_layer(
	const hnswlib::HierarchicalNSW<float>* index,
	const void* query,
	hnswlib::tableint entry,
	int layer) {

	hnswlib::tableint current = entry;
	float current_dist = index->fstdistfunc_(query, index->getDataByInternalId(current), index->dist_func_param_);

	bool changed = true;
	while (changed) {
		changed = false;
		int* data = (int*)index->get_linklist(current, layer);
		size_t size = index->getListCount((hnswlib::linklistsizeint*)data);
		hnswlib::tableint* datal = (hnswlib::tableint*)(data + 1);

		for (size_t j = 0; j < size; ++j) {
			hnswlib::tableint cand = *(datal + j);
			float d = index->fstdistfunc_(query, index->getDataByInternalId(cand), index->dist_func_param_);
			if (d < current_dist) {
				current_dist = d;
				current = cand;
				changed = true;
			}
		}
	}

	return {current, current_dist};
}

inline std::pair<hnswlib::tableint, float> closest_in_base_layer(
	const hnswlib::HierarchicalNSW<float>* index,
	const void* query,
	hnswlib::tableint entry,
	size_t ef) {

	auto top = index->template searchBaseLayerST<true>(entry, query, ef, nullptr, nullptr);
	hnswlib::tableint best = entry;
	float best_dist = index->fstdistfunc_(query, index->getDataByInternalId(best), index->dist_func_param_);
	while (!top.empty()) {
		auto p = top.top();
		top.pop();
		float d = p.first;
		if (d < best_dist) {
			best_dist = d;
			best = p.second;
		}
	}
	return {best, best_dist};
}

}  // namespace detail

// Computes the anomaly score following the actual HNSW/HENN top-down search path.
// The query point does NOT exist in the index; we just traverse the graph.
//
// p_L: start from entrypoint at top layer, greedy search within each layer.
// p_0: use base-layer ef search to get the closest candidate.
inline OutlierScoreResult query_outlier_score_path(
	const EndTreeIndex<float>& tree,
	const float* query,
	const std::vector<double>& level_weights = {},
	size_t ef_base_layer = 128) {

	if (!tree.index) {
		throw std::invalid_argument("query_outlier_score_path: tree.index is null");
	}
	if (query == nullptr) {
		throw std::invalid_argument("query_outlier_score_path: query is null");
	}
	if (tree.index->cur_element_count == 0) {
		throw std::invalid_argument("query_outlier_score_path: empty index");
	}

	const int max_level = tree.index->maxlevel_;
	OutlierScoreResult result;
	result.layers.resize(static_cast<size_t>(max_level + 1));

	// Start from the current enterpoint at top layer.
	hnswlib::tableint curr = tree.index->enterpoint_node_;
	float curr_dist = tree.index->fstdistfunc_((const void*)query, tree.index->getDataByInternalId(curr), tree.index->dist_func_param_);

	for (int layer = max_level; layer > 0; --layer) {
		auto best = detail::greedy_closest_in_layer(tree.index.get(), (const void*)query, curr, layer);
		curr = best.first;
		curr_dist = best.second;

		LayerOutlierScore& ls = result.layers[static_cast<size_t>(layer)];
		ls.layer = layer;
		ls.p = tree.index->getExternalLabel(curr);
		ls.dist_q_p = curr_dist;
	}

	// Base layer: do ef search from the last greedy point.
	{
		const size_t ef = std::max<size_t>(1, ef_base_layer);
		auto best0 = detail::closest_in_base_layer(tree.index.get(), (const void*)query, curr, ef);
		LayerOutlierScore& ls0 = result.layers[0];
		ls0.layer = 0;
		ls0.p = tree.index->getExternalLabel(best0.first);
		ls0.dist_q_p = best0.second;
	}

	// Compute per-layer anomaly score using parent stats (same math as query_outlier_score).
	constexpr double eps = 1e-12;
	constexpr double huge = 1e9;

	for (int layer = max_level - 1; layer >= 0; --layer) {
		LayerOutlierScore& ls = result.layers[static_cast<size_t>(layer)];
		const labeltype parent = result.layers[static_cast<size_t>(layer + 1)].p;
		if (parent == invalid_label || ls.p == invalid_label) {
			continue;
		}

		bool found_stats = false;
		ChildStats stats = detail::get_child_stats_or_default(tree.augmentation.parent_stats, parent, layer, &found_stats);

		ls.has_parent_stats = found_stats;
		ls.parent_coverage_radius = stats.coverage_radius;
		ls.parent_count = stats.count;
		ls.parent_mean = stats.mean;
		ls.parent_variance = stats.variance;

		const double dist = static_cast<double>(ls.dist_q_p);
		const double radius = std::max(static_cast<double>(stats.coverage_radius), eps);
		ls.score_dist = dist / radius;

		ls.score_density = (stats.count == 0) ? huge : (1.0 / static_cast<double>(stats.count));

		const double sigma = std::sqrt(std::max(stats.variance, eps));
		ls.score_struct = std::fabs(dist - stats.mean) / sigma;

		ls.s = ls.score_dist * ls.score_density * ls.score_struct;
		ls.weight = (static_cast<size_t>(layer) < level_weights.size()) ? level_weights[static_cast<size_t>(layer)] : 1.0;
		ls.weighted_s = ls.s * ls.weight;
		result.total_score += ls.weighted_s;
	}

	if (static_cast<size_t>(max_level) < result.layers.size()) {
		LayerOutlierScore& top = result.layers[static_cast<size_t>(max_level)];
		top.weight = (static_cast<size_t>(max_level) < level_weights.size()) ? level_weights[static_cast<size_t>(max_level)] : 1.0;
	}

	return result;
}

template <typename dist_t>
Augmentation augment_bottom_up(
	hnswlib::HierarchicalNSW<dist_t>* index,
	const float* points,
	size_t numPoints,
	size_t dim,
	hnswlib::SpaceInterface<dist_t>* space,
	const std::vector<int>& max_levels,
	size_t initial_search_k = 64,
	size_t max_search_k = 2048) {

	if (index == nullptr) {
		throw std::invalid_argument("augment_bottom_up: index is null");
	}
	if (points == nullptr) {
		throw std::invalid_argument("augment_bottom_up: points is null");
	}
	if (space == nullptr) {
		throw std::invalid_argument("augment_bottom_up: space is null");
	}
	if (max_levels.size() != numPoints) {
		throw std::invalid_argument("augment_bottom_up: max_levels size mismatch");
	}

	int max_level = 0;
	for (size_t i = 0; i < numPoints; ++i) {
		max_level = std::max(max_level, max_levels[i]);
	}

	Augmentation aug;
	aug.parent_of.assign(numPoints, detail::invalid_label);

	// For each "child layer" i, assign parents from layer i+1 (max_level >= i+1)
	for (int child_layer = 0; child_layer < max_level; ++child_layer) {
		detail::MinLevelFilter filter(&max_levels, child_layer + 1);

		for (labeltype u = 0; u < numPoints; ++u) {
			if (max_levels[u] != child_layer) continue;

			labeltype parent = detail::invalid_label;
			dist_t dist_to_parent{};

			// Try filtered ANN search first; if filter yields empty, fall back to exact scan.
			size_t search_k = std::max<size_t>(1, initial_search_k);
			std::vector<std::pair<dist_t, labeltype>> candidates;
			while (true) {
				candidates = index->searchKnnCloserFirst((const void*)(points + static_cast<size_t>(u) * dim), search_k, &filter);
				if (!candidates.empty()) break;
				if (search_k >= max_search_k || search_k >= numPoints) break;
				search_k = std::min(max_search_k, search_k * 2);
			}

			if (!candidates.empty()) {
				parent = candidates.front().second;
				dist_to_parent = candidates.front().first;
			} else {
				auto exact = detail::exact_nearest_parent(points, numPoints, dim, space, max_levels, u, child_layer + 1);
				parent = exact.first;
				dist_to_parent = exact.second;
			}

			if (parent == detail::invalid_label) {
				// This should only happen if there is no node in upper layers.
				continue;
			}

			aug.parent_of[u] = parent;
			aug.parent_stats[parent][child_layer].add(static_cast<double>(dist_to_parent));
		}
	}

	return aug;
}

// Builds HENN (same layer construction as henn.h) then augments epsilon nodes with bottom-up child stats.
inline EndTreeIndex<float> buildEndTree(
	float* data,
	int size,
	int dim,
	hnswlib::SpaceInterface<float>* space,
	int M,
	bool best = true,
	size_t initial_search_k = 64,
	size_t max_search_k = 2048) {

	auto layers_map = henn::buildBestWorstLayers(data, static_cast<size_t>(size), static_cast<size_t>(dim), space, static_cast<size_t>(M), best);

	std::vector<int> max_levels(static_cast<size_t>(size), 0);
	for (const auto& kv : layers_map) {
		int idx = kv.first;
		int lvl = kv.second;
		if (idx >= 0 && idx < size) {
			max_levels[static_cast<size_t>(idx)] = lvl;
		}
	}

	EndTreeIndex<float> result;
	result.max_levels = std::move(max_levels);
	result.index.reset(henn::buildHENN<float>(layers_map, data, static_cast<size_t>(size), static_cast<size_t>(dim), space));
	result.augmentation = augment_bottom_up<float>(
		result.index.get(),
		data,
		static_cast<size_t>(size),
		static_cast<size_t>(dim),
		space,
		result.max_levels,
		initial_search_k,
		max_search_k);

	return result;
}

}  // namespace endtree
