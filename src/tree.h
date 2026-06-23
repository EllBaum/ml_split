// tree.h — Unrooted binary tree as flat arrays.
// Nodes: 0..n_taxa-1 leaves, n_taxa..2*n_taxa-3 internal.
//
// Branch lengths are NOT stored in the tree. They are maintained externally
// as a flat array of size n_nodes*3, parallel to the neighbors array:
//   bl[u*3 + k] = branch length to neighbors[u*3+k]
// This gives O(n) space and O(1) lookup (scan at most 3 slots).
//
// Static helpers bl_get / bl_set / make_bl operate on external arrays.
// to_newick accepts an optional const double* bl (size n_nodes*3) or nullptr.
// from_newick returns the Tree by value and writes branch lengths into bl_out
// (pre-allocated, size n_nodes*3).
//
// spr_move and nni_move have an in_place argument (default true).
//   in_place=true:  modifies this tree topology in place, returns void.
//   in_place=false: deep-copies tree, applies move to copy, returns new Tree.

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <random>
#include <cassert>
#include <iostream>

constexpr int32_t INVALID = -1;

struct PruneResult;  // defined after Tree (holds a Tree by value)

class Tree {
public:
    int n_taxa  = 0;
    int n_nodes = 0;   // 2*n_taxa - 2
    int n_edges = 0;   // 2*n_taxa - 3  (useful constant, no edge arrays)

    std::vector<std::string> taxon_names;
    std::vector<int32_t>     neighbors;   // size: n_nodes * 3

    // ── Construction ──────────────────────────────────────────────────────────

    Tree() = default;
    Tree(int n_taxa, const std::vector<std::string>& taxon_names);

    static Tree random_tree(const std::vector<std::string>& taxa,
                            std::mt19937& rng);

    // Parse Newick string. bl_out must be pre-allocated to n_nodes*3 doubles.
    // Branch lengths are written into bl_out (default 1.0 if absent in string).
    // If taxa_order is empty, extracts taxon names from the string.
    static Tree from_newick(const std::string& newick,
                            double* bl_out,
                            const std::vector<std::string>& taxa_order = {});

    // ── Branch length helpers (external bl array, size n_nodes*3) ─────────────

    // Get branch length for edge (u,v). O(1): scans at most 3 slots.
    static double bl_get(const std::vector<int32_t>& neighbors,
                         const double* bl, int u, int v);

    // Set branch length for edge (u,v) symmetrically.
    static void   bl_set(const std::vector<int32_t>& neighbors,
                         double* bl, int u, int v, double value);

    // Allocate and fill a bl array of size n_nodes*3 with a default value.
    static std::vector<double> make_bl(int n_nodes, double default_val = 0.1);

    // ── Query helpers ─────────────────────────────────────────────────────────

    bool is_leaf(int node) const { return node < n_taxa; }
    int  degree(int node)  const;
    bool are_connected(int u, int v) const;

    // fills out[], returns count
    int neighbors_of    (int node,                   int out[3]) const;
    int neighbors_except(int node, int exclude,      int out[2]) const;
    int other_neighbor  (int node, int excl1, int excl2)         const;

    // ── Traversal ─────────────────────────────────────────────────────────────

    std::vector<int> postorder(int start, int exclude = INVALID) const;

    // ── SPR enumeration ───────────────────────────────────────────────────────

    // All valid (prune_u, prune_v) pairs. prune_u is always internal.
    // Only pairs where the remaining tree has at least one valid regraft
    // candidate are included.
    std::vector<std::pair<int,int>> spr_prune_points() const;

    // All valid (ra, rb) regraft candidates for prune at (prune_u, prune_v),
    // up to radius. ra is closer to prune_u, rb is the far side.
    // Excludes edges adjacent to prune_u (depth 0). DFS order.
    std::vector<std::pair<int,int>> spr_candidates(int prune_u, int prune_v,
                                                    int radius = 10) const;

    // ── Topology moves ────────────────────────────────────────────────────────

    // SPR move. prune_u = sticky end (internal), prune_v = subtree base.
    // in_place=true (default): modifies this tree, returns empty Tree.
    // in_place=false: returns a new Tree with the move applied.
    //
    // Branch length convention for external bl array (handled by caller):
    //   bl_new[nb1, nb2] = bl_orig[prune_u, nb1]
    //   bl_new[prune_u, ra] = bl_orig[ra, rb]
    //   bl_new[prune_u, rb] = bl_orig[prune_u, nb2]
    //   bl_new[prune_u, pv] = unchanged
    //
    // WARNING: in_place=true modifies topology immediately. Copy first if needed.
    Tree spr_move(int prune_u, int prune_v,
                  int regraft_a, int regraft_b,
                  bool in_place = true);

    // NNI on internal edge (edge_u, edge_v). swap_which = 0 or 1.
    // NNI is equivalent to SPR with radius=1.
    // in_place=true (default): modifies this tree, returns empty Tree.
    // in_place=false: returns a new Tree with the move applied.
    Tree nni_move(int edge_u, int edge_v,
                  int swap_which = 0,
                  bool in_place  = true);

    // ── Newick I/O ────────────────────────────────────────────────────────────

    // Write tree as unrooted Newick string.
    // bl: optional flat array of size n_nodes*3 parallel to neighbors.
    //     Pass nullptr to omit branch lengths.
    std::string to_newick(const double* bl = nullptr) const;

    // ── Leaf pruning + anchor recording ───────────────────────────────────────

    // Remove the named leaves, suppress the resulting degree-2 nodes (summing
    // their two branch lengths), and renumber the survivors into a smaller Tree.
    //
    // bl: external branch-length array for THIS tree (size n_nodes*3), or nullptr
    //     to use 0.1 everywhere (branch lengths in the result are then all 0.1).
    //
    // Likelihood-exact: dropping a pendant leaf and summing the two through
    // branches of its (former) neighbor leaves the remaining tree's likelihood
    // unchanged, because P(t1)·P(t2) = P(t1+t2).
    //
    // For each removed leaf (in the order given) the result records the EDGE in
    // the pruned tree that the leaf collapsed onto — the "anchor" edge, returned
    // as a (u,v) node pair in the RENUMBERED result tree. Leaves strung along the
    // same surviving branch share one anchor edge; leaves in different regions get
    // different anchors. This is the pseudoroot edge a merge would root on.
    //
    // Throws if a name is not a leaf of this tree, or if fewer than 3 leaves
    // would remain.
    PruneResult pruned(const std::vector<std::string>& remove_leaf_names,
                       const double* bl) const;

    // Build a renumbered Tree from an explicit edge list over arbitrary integer
    // tokens. `leaf_tokens` maps each leaf token to its taxon name; any token not
    // in the map is treated as an internal node. Leaves are numbered 0..k-1
    // sorted by name, internals follow. bl_out is resized to n_nodes*3 and filled.
    // The edge list must describe a valid unrooted binary tree (≥3 leaves).
    static Tree assemble(const std::vector<std::tuple<int,int,double>>& edges,
                         const std::unordered_map<int,std::string>& leaf_tokens,
                         std::vector<double>& bl_out);

    // ── Validation ────────────────────────────────────────────────────────────

    bool validate() const;

    // ── Debug ─────────────────────────────────────────────────────────────────

    void print_adjacency() const;

private:
    void add_neighbor   (int node, int neighbor);
    void remove_neighbor(int node, int neighbor);
    void attach         (int u, int v);
    void detach         (int u, int v);

    std::string subtree_newick(int node, int exclude,
                               const double* bl) const;
};

// Result of Tree::pruned. `tree` is the renumbered smaller tree; `bl` is its
// external branch-length array (size tree.n_nodes*3, parallel to tree.neighbors).
// `anchors[i]` is the edge (u,v) in `tree` that removed leaf `removed_names[i]`
// collapsed onto. anchors / removed_names are parallel to the input name list.
struct PruneResult {
    Tree                             tree;
    std::vector<double>              bl;
    std::vector<std::pair<int,int>>  anchors;
    std::vector<std::string>         removed_names;
};
