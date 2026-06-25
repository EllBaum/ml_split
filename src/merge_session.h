// merge_session.h -- outgroup-based subtree merge.
//
// Joins two subtrees (each carrying outgroups) at a connector, choosing the best
// of a 14x14 window of attachment-edge pairs by 5-branch-optimized likelihood,
// then restores the inherited outgroups. Returns (merged_newick, loglik).
//
// The two outgroups-of-interest are removed (their recorded anchors seed the two
// windows) and NOT restored -- each is a real leaf already present in the other
// subtree. Inherited outgroups are removed for the search and restored after,
// pointer-style (only which edge they sit on matters).

#pragma once

#include "tree.h"
#include "msa.h"
#include "subst_model.h"
#include "merge_prep.h"
#include <string>
#include <vector>
#include <utility>

// Row-slice an MSA to `names` (in this order), keeping the same pattern columns
// and weights. Both slices of one MSA stay column-aligned, which is what lets the
// connector multiply the two subtrees' CLVs like-for-like.
MSA slice_msa(const MSA& full, const std::vector<std::string>& names);

// Up to `n` edges of `t` nearest the anchor edge (a0,a1), inclusive, by BFS over
// the edge adjacency. Returned as (u,v) node pairs in `t`. This is one subtree's
// merge window.
std::vector<std::pair<int,int>> window_edges(const Tree& t, int a0, int a1,
                                             int n = 14);

// -- Merge ---------------------------------------------------------------------

struct MergeInput {
    std::string newick_a, newick_b;
    std::string interest_a, interest_b;            // outgroup-of-interest per side
    // Inherited outgroups, pre-grouped: each inner vector is one clade (a single
    // scattered leaf is a clade of size 1). Removed for the search, restored after.
    std::vector<std::vector<std::string>> inherited_a, inherited_b;
};

struct MergeResult {
    std::string newick;     // merged tree over realA U realB (+ restored inherited)
    double      loglik;     // 5-BLO-optimized joined LL over realA U realB
};

// Merge two subtrees. `full` must contain rows for at least realA U realB (the
// search taxa); it is sliced per side. Returns the best of the window?window
// attachment candidates plus the inherited outgroups restored in place.
MergeResult run_merge(const MergeInput& in, const MSA& full, SubstModel& model,
                      int window = 14, double connector_init = 0.1,
                      const std::string& full_blo = "off");

// Split-edge choice: when an inherited clade's anchor is the edge the connector
// split (now two sub-edges (eu,mid) and (mid,ev) sharing connector node `mid`),
// attach the WHOLE clade on each sub-edge, score, and return the better sub-edge
// as an (a,b) pair. No BLO; the clade's internal topology/BLs stay frozen. Needs
// the clade's sequences, so `full` must include the clade's taxa. `clade` tokens
// must already be offset to be unique within `m_search`.
std::pair<int,int> choose_split_subedge(const AdjTree& m_search,
                                        const DetachedClade& clade,
                                        int eu, int mid, int ev,
                                        const MSA& full, SubstModel& model);

