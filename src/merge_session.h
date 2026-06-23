// merge_session.h — outgroup-based subtree merge.
//
// Joins two subtrees (each carrying outgroups) at a connector, choosing the best
// of a 14x14 window of attachment-edge pairs by 5-branch-optimized likelihood,
// then restores the inherited outgroups. Returns (merged_newick, loglik).
//
// The two outgroups-of-interest are removed (their recorded anchors seed the two
// windows) and NOT restored — each is a real leaf already present in the other
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
