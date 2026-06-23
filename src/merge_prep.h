// merge_prep.h — inherited-outgroup detach/regraft over a dynamic adjacency.
//
// An inherited outgroup (single scattered leaf, or a whole clade) is removed
// from a subtree before the merge likelihood search, then put back in place
// afterward. detach_clade captures the clade WITH its branch lengths and the
// anchor edge it collapses onto; regraft_clade restores it exactly (or, for the
// merge split case, onto a chosen sub-edge — see regraft_clade_on).
//
// AdjTree is a dynamic adjacency (nodes can be added on graft), built from a
// Tree and materialized back to a renumbered Tree via Tree::assemble.

#pragma once

#include "tree.h"
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

struct AdjTree {
    std::map<int, std::vector<std::pair<int,double>>> adj;  // node -> (neighbor, bl)
    std::map<int, std::string>                        leaf_name;  // leaf node -> name
    int fresh = 0;                                          // next fresh internal id

    static AdjTree from(const Tree& t, const double* bl);

    bool   is_leaf(int u) const { return leaf_name.count(u) > 0; }
    int    degree (int u) const { return (int)adj.at(u).size(); }
    int    new_internal()       { return fresh++; }
    double get_bl (int u, int v) const;
    void   connect(int u, int v, double w);
    void   disconnect(int u, int v);
    void   erase_node(int u) { adj.erase(u); }

    // Renumber surviving nodes into a Tree (leaves sorted by name) + bl array.
    Tree to_tree(std::vector<double>& bl_out) const;
};

// A detached clade captured with full branch lengths.
struct DetachedClade {
    std::vector<std::tuple<int,int,double>> edges;     // clade-internal adjacency + bl
    std::map<int,std::string>               token_name;// clade leaf tokens -> name
    int                                     connect_token = INVALID;  // clade root
    double                                  stem_bl  = 0.0;           // host-side stem bl
    std::pair<int,int>                      anchor{INVALID,INVALID};  // host edge collapsed onto
    double                                  half_a = 0.0, half_b = 0.0; // original split
};

// Find (P, c): the host-side node P and clade-side node c whose leaves below c
// (away from P) are exactly `clade`. Throws if `clade` is not a clade.
std::pair<int,int> find_stem(const AdjTree& a, const std::set<std::string>& clade);

// Detach the named clade, suppressing the host node; returns the capture.
DetachedClade detach_clade(AdjTree& a, const std::set<std::string>& clade);

// Restore a detached clade exactly onto its recorded anchor edge (original split).
void regraft_clade(AdjTree& a, const DetachedClade& d);

// Restore onto a specific host edge (edge_u, edge_v) — used when the merge split
// the original anchor edge and the side is chosen by likelihood. The edge is
// split at `frac` of its length (frac in (0,1)); branch lengths are not
// load-bearing for an inherited outgroup (it is only a pointer next round).
void regraft_clade_on(AdjTree& a, const DetachedClade& d,
                      int edge_u, int edge_v, double frac = 0.5);
