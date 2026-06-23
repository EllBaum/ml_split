// likelihood_scorer_core.cpp: Constructor, allocation, SIMD dispatch.

#include "likelihood_scorer.h"
#include "prof.h"
#include "counters.h"

// Instrumentation-only: thread-local NR iter snapshot. Set by
// _nr_branch_with_clvs_with_status and _nr_branch_with_sumtable on every
// call. Read by try_add to histogram B1/B2/B3 iter counts.
namespace { thread_local int g_last_nr_iters = 0; }
#include <algorithm>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>  // AVX2 SIMD
#include <queue>
#include <set>
#include <stack>
#include <stdexcept>
#include <string>
#include <cmath>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>

static constexpr double SCALE_THRESHOLD  = 1e-40;
static constexpr double SCALE_FACTOR     = 1e40;
static const     double LOG_SCALE        = -std::log(1e40); // negative: correction for inflation

// ================================================================== //
// LikelihoodScorer — constructor                                      //
// ================================================================== //

LikelihoodScorer::LikelihoodScorer(Tree& tree, const MSA& msa, SubstModel& model,
                                   const double* bl)
    : tree_(&tree), msa_(&msa), model_(&model),
      K_(model.K()), L_(msa.n_patterns), n_(tree.n_nodes),
      score_(0.0)
{
    assert(tree.n_taxa == msa.n_taxa);
    assert(tree.taxon_names == msa.taxon_names);

    // Branch lengths from external flat n_nodes*3 array, or default 0.1
    bl_.assign(n_ * n_, 0.0);
    if (bl != nullptr) {
        for (int u = 0; u < n_; u++) {
            int base = u * 3;
            for (int k = 0; k < 3; k++) {
                int v = tree.neighbors[base + k];
                if (v != INVALID) {
                    bl_[u * n_ + v] = bl[base + k];
                }
            }
        }
    } else {
        for (int u = 0; u < n_; u++) {
            int nbs[3];
            int nc = tree.neighbors_of(u, nbs);
            for (int i = 0; i < nc; i++) {
                bl_[u * n_ + nbs[i]] = 0.1;
            }
        }
    }

    // Allocate arrays
    tip_clv_.assign(tree.n_taxa * K_ * L_, 0.0);
    clv_arr_.assign((size_t)(n_ - tree.n_taxa) * 3 * K_ * L_, 0.0);
    log_scale_arr_.assign(n_ * 3 * L_, 0.0);
    p_arr_.assign(n_ * 3 * K_ * K_, 0.0);
    dirty_.assign(n_ * 3, 1);  // start dirty; _full_recompute clears them
    toward_.assign(n_, INVALID);

    // Flat bitset for roll_candidates_dedup duplicate detection.
    // Sized n_*n_ (max key is (n-1)*n + (n-1)). All zeros + gen=1 means
    // "no entries seen yet". Generation counter is initialized in the
    // header (= 1); we only need to size the buffer here.
    seen_radius1_.assign((size_t)n_ * (size_t)n_, 0u);

    // calc_start = leaf0's single neighbor
    {
        int out[3];
        tree.neighbors_of(0, out);
        calc_start_ = out[0];
    }

    _build_tip_clvs();
    _full_recompute();
}

// ================================================================== //
// SIMD dispatch: detect AVX-512 availability at startup so the same    //
// binary runs on heterogeneous cluster nodes (some Skylake-X with     //
// AVX-512, some older without). The check happens once and the        //
// hot-path branch is well-predicted across calls.                     //
// ================================================================== //