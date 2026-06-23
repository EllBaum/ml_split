// likelihood_scorer.h - Maximum likelihood scorer using clv(node, excl).

#pragma once

#include "tree.h"
#include "msa.h"
#include "subst_model.h"
#include "jck.h"
#include "empirical_aa_model.h"
#include "jtt.h"
#include "gtr.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <cstdint>
#include <cmath>
#include <limits>

// Env vars (default OFF): SIMD_VERBOSE prints the SIMD path; CLV_VERIFY checks CLVs.

struct LikelihoodSPRCandidate {
    double log_likelihood = 0.0;
    int prune_u = INVALID, prune_v = INVALID;
    int ra      = INVALID, rb      = INVALID;
    int nb1     = INVALID, nb2     = INVALID;

    std::vector<int>    path_nodes;   // prune_u..ra, excl. prune_u

    // Optimized regraft BLs; NaN = use split/merge convention. commit() writes
    // whichever are non-NaN.
    double triplet_bl_pendant = std::numeric_limits<double>::quiet_NaN();
    double triplet_bl_ra      = std::numeric_limits<double>::quiet_NaN();
    double triplet_bl_rb      = std::numeric_limits<double>::quiet_NaN();
    double bl_ra_nb_other     = std::numeric_limits<double>::quiet_NaN();
    double bl_ra_ra_other     = std::numeric_limits<double>::quiet_NaN();
};


class LikelihoodScorer {
public:
    // msa must be built with parsimony_informative_only=false (constant sites
    // contribute). bl: flat n_nodes*3 parallel to tree.neighbors, else 0.1.
    LikelihoodScorer(Tree& tree, const MSA& msa, SubstModel& model,
                     const double* bl = nullptr);

    double score() const { return score_; }

    // Every _clv materialization (memcpy-reuse counts; _clv_view excluded). run_ml budget.
    long clv_calc_count() const { return clv_calc_count_; }

    double branch_length(int u, int v) const { return bl_[u * n_ + v]; }
    std::vector<double> get_bl_array() const;   // flat n_nodes*3

    // Top-k SPR candidates in [radius_min, radius], best first. thorough=false:
    // fixed-BL (FAST); true: triplet-BLO, 5-BLO at depth==0 (SLOW). Each
    // (prune_u, ra) center scored once per round (reset on commit). only_ra set:
    // replay just (only_ra, only_rb), bypassing and not touching dedup state.
    std::vector<LikelihoodSPRCandidate> roll_candidates_dedup(
        int prune_u, int prune_v,
        int radius = 10, int k = 1,
        bool thorough = false,
        int radius_min = 1,
        int only_ra = INVALID, int only_rb = INVALID,
        double triplet_eps = 1000.0, double fiveblo_eps = 1000.0) const;

    void commit(const LikelihoodSPRCandidate& cand);

    // ── Merge hooks (used by MergeSession; this is a separate codebase) ────────

    int n_states()   const { return K_; }
    int n_patterns() const { return L_; }

    // Directional CLV at `node` looking away from `excl` (and its log-scale),
    // over this scorer's pattern set. After construction the tree's CLVs are
    // materialized and clean, so for a frozen subtree this is a cheap cached
    // read — the rigid-subtree boundary condition for a merge.
    void clv(int node, int excl,
             std::vector<double>& clv_out, std::vector<double>& ls_out) const {
        clv_out.resize((size_t)K_ * L_);
        ls_out.resize(L_);
        _clv(node, excl, clv_out.data(), ls_out.data());
    }

    // Join two frozen subtrees at a fresh connector and optimize the 5 branches
    // around it. Boundary CLVs are the directional CLVs at the two anchor edges:
    // A side (pv, rb) from scorer_A, B side (nb, ro) from scorer_B — all over the
    // same (sliced) pattern set. init_bl5 = bl_out order =
    //   {connector, newA-pv, newA-rb, newB-nb, newB-ro}.
    // Call on either sub-scorer (model/K/L/weights are shared). Returns the
    // optimized joined log-likelihood; bl_out[5] filled in the same order.
    double optimize_five_for_merge(
        const double* pv_clv, const double* pv_ls,
        const double* rb_clv, const double* rb_ls,
        const double* nb_clv, const double* nb_ls,
        const double* ro_clv, const double* ro_ls,
        const double init_bl5[5], double bl_out[5],
        double eps = 0.1) const
    {
        // Dummy node ids: with all 4 CLVs supplied and init_bl5 set, the ids are
        // never used to index tree_/bl_ (no _clv fallback, no bl_ reads).
        return _optimize_five_branches_at_nni(
            /*prune_u=*/0, /*prune_v=*/1, /*ra=*/2, /*rb=*/3,
            /*nb_other=*/4, /*ra_other=*/5, bl_out,
            pv_clv, pv_ls, rb_clv, rb_ls,
            ro_clv, ro_ls, nb_clv, nb_ls,
            eps, init_bl5);
    }

    // Let a later thorough pass re-examine centers the FAST walk stamped.
    void reset_dedup_seen() { _clear_seen_radius1(); }

    // Returns final log-likelihood. Modes: fast (CLVs once/sweep, stop at
    // lh_epsilon), thorough (CLVs after each branch), fixed (exactly n_sweeps),
    // lbfgsb (TODO -> thorough).
    double optimize_branch_lengths(const std::string& mode = "fast",
                                   double lh_epsilon        = 10.0,
                                   int    max_sweeps        = 32,
                                   int    n_sweeps          = 1,
                                   double blmin             = 1e-6,
                                   double blmax             = 100.0,
                                   const LikelihoodSPRCandidate* cand = nullptr);

private:
    Tree*        tree_;
    const MSA*   msa_;
    SubstModel*  model_;

    int K_;
    int L_;
    int n_;

    int    calc_start_;
    double score_;

    std::vector<double>  bl_;        // flat n*n, symmetric
    mutable std::vector<double>  tip_clv_;

    // Compact tips for the tip kernels: per-pattern state code + code->bitmask
    // table (gap remaps to all-bits-set, matching tip_clv_'s all-1.0 gap).
    std::vector<uint8_t>  tip_codes_;
    std::vector<uint32_t> tip_mask_table_;
    int                   n_tip_codes_ = 0;

    // Directional CLVs: slot s at node v looks AWAY from neighbors[v*3+s]
    // (leaves use slot 0). p_arr_/log_scale_arr_ index alike; dirty_==0 means
    // the slot is in sync with bl_ and its two non-excl children.
    mutable std::vector<double>  clv_arr_;        // inner nodes only (N-2), 3 slots each
    std::vector<double>  log_scale_arr_;
    std::vector<double>  p_arr_;
    std::vector<uint8_t> dirty_;

    inline int _slot_of(int v, int nbr) const {
        const int32_t* nbrs = &tree_->neighbors[v * 3];
        if (nbrs[0] == nbr) return 0;
        if (nbrs[1] == nbr) return 1;
        if (nbrs[2] == nbr) return 2;
        return -1;
    }

    // CLV slot pointer. clv_arr_ holds only the N-2 inner nodes
    // (3 directional slots each); a leaf has one neighbor so it only needs slot
    // 0, which lives in tip_clv_. The recursion already routes is_leaf reads to
    // tip_clv_; this also covers child/c_other reads of a leaf's slot 0. Writes
    // only ever target inner nodes.
    inline double* _clv_slot(int node, int s) const {
        return (node < tree_->n_taxa)
                 ? &tip_clv_[(size_t)node * (K_ * L_)]
                 : &clv_arr_[((size_t)(node - tree_->n_taxa) * 3 + s) * (K_ * L_)];
    }

    std::vector<int32_t> toward_;

    void _scale_clv_inplace(double* clv, double* log_scale) const;

    // Kernels: P @ clv -> out, state-major. _mul also multiplies by mul[];
    // _pair returns (P1@clv1)*(P2@clv2); _tip variants take a sparse leaf CLV
    // via the (taxon, t) lookup table instead of a dgemv.
    void _p_apply(const double* P, const double* clv, double* out) const;
    void _p_apply_mul(const double* P, const double* clv,
                      const double* mul, double* out) const;
    void _p_apply_pair(const double* P1, const double* clv1,
                       const double* P2, const double* clv2,
                       double* out) const;
    void _p_apply_tip(const double* P, double t, int taxon, double* out) const;
    void _p_apply_mul_tip(const double* P, double t, int taxon,
                          const double* mul, double* out) const;
    void _p_apply_tip_internal_pair(const double* P_tip, double t_tip, int taxon_tip,
                                    const double* P_int, const double* clv_int,
                                    double* out) const;
    void _p_apply_tip_tip_pair(const double* P1, double t1, int taxon1,
                               const double* P2, double t2, int taxon2,
                               double* out) const;

    mutable std::unordered_map<int,
        std::unordered_map<double, std::vector<double>>> tip_lookup_cache_;

    void _clv_mul(const double* a, const double* b, double* out) const;

    void _clv(int node, int excl, double* out_clv, double* out_log_scale) const;
    // Zero-copy for toward/leaf; otherwise fills scratch and returns it.
    void _clv_view(int node, int excl,
                   const double** out_clv_ptr, const double** out_ls_ptr,
                   double* scratch_clv, double* scratch_ls) const;
    void _p_for_edge(int u, int v, double* out) const;

    double _pseudoroot_loglik() const;
    double _total_loglik(const double* root_clv, const double* log_scale) const;
    void _build_tip_clvs();
    void _refresh_toward();
    void _propagate_up_ll(int start, std::unordered_set<int>& visited);
    void _full_recompute();

    // CLV_VERIFY only; recursive ground truth, not hot-path-safe.
    void _clv_truth(int node, int excl, double* out_clv, double* out_ls) const;
    void _verify_clv_truth() const;

    double _optimize_single_branch(int u, int v,            // writes bl_ in place
                                   double blmin = 1e-6, double blmax = 100.0,
                                   double tol = 1e-7, int max_iter = 30);
    double _optimize_triplets(double lh_epsilon, int max_sweeps,   // mode="triplet"
                              double blmin, double blmax,
                              double tol = 1e-7, int max_iter = 30);

    // NR on one branch from flanking CLVs A, B; returns BL, does not write bl_.
    // _with_status reports convergence so callers can revert to t_init.
    double _nr_branch_with_clvs(const double* A, const double* B,
                                double t_init,
                                double blmin, double blmax,
                                double tol, int max_iter) const;
    double _nr_branch_with_clvs_with_status(const double* A, const double* B,
                                            double t_init,
                                            double blmin, double blmax,
                                            double tol, int max_iter,
                                            bool* converged) const;

    // Sumtable BLO: S built once per branch, then NR reads it; Lpi_mat_ folds pi in.
    mutable std::vector<double> sumtable_main_;
    mutable std::vector<double> sumtable_X_;
    mutable std::vector<double> sumtable_Z_;
    mutable std::vector<double> Lpi_mat_;
    mutable bool                Lpi_mat_built_ = false;
    void _build_sumtable(const double* A, const double* B, double* S_out) const;
    double _nr_branch_with_sumtable(const double* S, double t_init,
                                    double blmin, double blmax,
                                    double tol, int max_iter,
                                    bool* converged = nullptr) const;

    // 5-branch BLO at the post-SPR center (pu, ra), radius-1 thorough only.
    // bl_out = [pu-ra, pu-pv, pu-rb, ra-nb_other, ra-ra_other]. The 4 boundary
    // CLVs (pv, rb, ra_other, nb_other) may be passed to skip their _clv();
    // nullptr computes them.
    double _optimize_five_branches_at_nni(
        int prune_u, int prune_v, int ra, int rb,
        int nb_other, int ra_other,
        double bl_out[5],
        const double* pv_clv_in = nullptr, const double* pv_ls_in = nullptr,
        const double* rb_clv_in = nullptr, const double* rb_ls_in = nullptr,
        const double* ro_clv_in = nullptr, const double* ro_ls_in = nullptr,
        const double* nb_clv_in = nullptr, const double* nb_ls_in = nullptr,
        double eps_5blo_in = 1000.0,
        // Merge hook: when non-null, the 5 initial branch lengths are taken from
        // init_bl5 (order = bl_out) instead of read from bl_ by node id. Lets the
        // merge drive this with hypothetical connector nodes that aren't in tree_.
        const double* init_bl5 = nullptr
    ) const;

    // n*n stamps vs a generation counter -> O(1) clear (memset only on wrap).
    mutable std::vector<uint32_t> seen_radius1_;
    mutable uint32_t              seen_radius1_gen_ = 1;
    mutable long                  clv_calc_count_ = 0;

    inline void _clear_seen_radius1() const {
        if (++seen_radius1_gen_ == 0u) {
            std::fill(seen_radius1_.begin(), seen_radius1_.end(), 0u);
            seen_radius1_gen_ = 1u;
        }
    }
};