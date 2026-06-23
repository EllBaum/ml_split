// likelihood_scorer_roll.cpp — part of the likelihood_scorer_unit.cpp umbrella.

#include "likelihood_scorer.h"
#include <cstdlib>
#ifdef __AVX2__
#include <immintrin.h>
#endif

// FUSED_SCORE (env, default off): fold pi into the pendant contribution once
// per prune (wpv = π⊙pv_contrib) and score each no-BLO candidate with one dot
// product instead of clv_mul+scale+π-sum. Bit-identical (both operands are
// already O(1)-scaled, so the dropped final scale never fires). #TDL - remove safely.
static bool _fused_score_enabled() {
    static const bool v = [] {
        const char* e = std::getenv("FUSED_SCORE");
        return e && e[0] == '1';
    }();
    return v;
}

// ================================================================== //
// roll_candidates_dedup — FRAME-level (pu,ra) dedup + 5-BLO at depth==0; //
// depth>=1 keeps triplet behaviour.                                     //
// ================================================================== //

std::vector<LikelihoodSPRCandidate>
LikelihoodScorer::roll_candidates_dedup(int prune_u, int prune_v,
                                   int radius, int k,
                                   bool thorough,
                                   int radius_min,
                                   int only_ra, int only_rb,
                                   double triplet_eps, double fiveblo_eps) const {
    PROF_SCOPE("roll_candidates_dedup");
    // Enumeration counters: rcd.calls_* = invocations per phase; rcd.radius_*
    // mean = effective radius (constant for fast, sliding for slow).
    COUNTER_INC(thorough ? "rcd.calls_slow" : "rcd.calls_fast");
    COUNTER_ADD(thorough ? "rcd.radius_slow" : "rcd.radius_fast", radius);
    const Tree& tree = *tree_;
    int KL = K_ * L_, KK = K_ * K_;
    const double* pi = model_->pi();

    // nb1, nb2: same ordering as spr_move
    int nb1_nb2[2];
    tree.neighbors_except(prune_u, prune_v, nb1_nb2);
    int nb1 = nb1_nb2[0], nb2 = nb1_nb2[1];

    // Post-SPR BL convention: merged nb1—nb2 edge = bl[nb1,pu]+bl[nb2,pu] (sum);
    // the two new pu sub-edges each = bl[ra,rb]/2 (split). P_merged depends only
    // on the prune point; the sub-edge P depends on bl[ra,rb] (per-candidate, in
    // try_add). Scratch buffers below are static thread_local + resize_if_smaller
    // so they reuse memory across calls instead of heap-allocating each time.
    auto resize_if_smaller = [](std::vector<double>& v, size_t n) {
        if (v.size() < n) v.assign(n, 0.0);
    };
    static thread_local std::vector<double> P_merged;
    resize_if_smaller(P_merged, KK);
    model_->p_matrix(bl_[nb1 * n_ + prune_u] + bl_[nb2 * n_ + prune_u],
                     P_merged.data());

    // pv contribution (P-applied toward pu, unchanged — pendant edge)
    static thread_local std::vector<double> pv_raw, pv_ls, P_pu_pv, pv_contrib;
    resize_if_smaller(pv_raw, KL);
    resize_if_smaller(pv_ls, L_);
    resize_if_smaller(P_pu_pv, KK);
    resize_if_smaller(pv_contrib, KL);
    _clv(prune_v, prune_u, pv_raw.data(), pv_ls.data());
    _p_for_edge(prune_u, prune_v, P_pu_pv.data());
    _p_apply(P_pu_pv.data(), pv_raw.data(), pv_contrib.data());
    // pv_ls unchanged: P-apply doesn't affect scale

    // FUSED_SCORE precompute: wpv = π ⊙ pv_contrib (once per prune point).
    const bool fused_score = _fused_score_enabled();
    static thread_local std::vector<double> wpv;
    if (fused_score) {
        resize_if_smaller(wpv, KL);
        for (int i = 0; i < K_; ++i) {
            const double pii = pi[i];
            const double* src = pv_contrib.data() + (size_t)i * L_;
            double* dst = wpv.data() + (size_t)i * L_;
            for (int l = 0; l < L_; ++l) dst[l] = pii * src[l];
        }
    }

    // Max-heap: (score, counter, candidate)
    struct HeapEntry {
        double score; int ctr; LikelihoodSPRCandidate cand;
        bool operator<(const HeapEntry& o) const { return score > o.score; } // min-heap on ll
    };
    std::vector<HeapEntry> heap;
    heap.reserve(k + 1);
    int ctr = 0;

    // Pre-allocated path buffers (overwritten per DFS step, copied on push)
    int max_d = tree.n_nodes;
    static thread_local std::vector<int>    stk_nodes;
    static thread_local std::vector<double> stk_clvs;
    static thread_local std::vector<double> stk_log_scales;
    if ((int)stk_nodes.size()      < max_d)      stk_nodes.assign(max_d, 0);
    if ((int)stk_clvs.size()       < max_d * KL) stk_clvs.assign(max_d * KL, 0.0);
    if ((int)stk_log_scales.size() < max_d * L_) stk_log_scales.assign(max_d * L_, 0.0);

    // Scratch buffers reused across candidates
    static thread_local std::vector<double> P_half, rb_raw, rb_ls,
                                            contrib_ra, contrib_rb,
                                            pseudo_clv, pseudo_ls,
                                            final_clv, final_ls,
                                            site_lik, tmp_P;
    resize_if_smaller(P_half, KK);
    resize_if_smaller(rb_raw, KL);
    resize_if_smaller(rb_ls,  L_);
    resize_if_smaller(contrib_ra, KL);
    resize_if_smaller(contrib_rb, KL);
    resize_if_smaller(pseudo_clv, KL);
    resize_if_smaller(pseudo_ls,  L_);
    resize_if_smaller(final_clv,  KL);
    resize_if_smaller(final_ls,   L_);
    resize_if_smaller(site_lik,   L_);
    resize_if_smaller(tmp_P,      KK);

    // Triplet-BLO scratch (used inside try_add). pv_contrib_local: per-candidate
    // copy of pv_contrib that triplet BLO mutates when optimizing pendant B1.
    // A_pu_ra/A_pu_rb: CLVs at pu toward ra/rb, the "A" input for the B2/B3 NR.
    static thread_local std::vector<double> pv_contrib_local, A_pu_ra, A_pu_rb;
    static thread_local std::vector<double> P_B1, P_B2, P_B3;
    resize_if_smaller(pv_contrib_local, KL);
    resize_if_smaller(A_pu_ra, KL);
    resize_if_smaller(A_pu_rb, KL);
    resize_if_smaller(P_B1, KK);
    resize_if_smaller(P_B2, KK);
    resize_if_smaller(P_B3, KK);

    // Triplet BLO config: 32 outer sweeps, loose outer eps (default 1000 ->
    // ~1 sweep), tight inner NR (tol 1e-7, 30 iters). The looseness is in the
    // outer eps, not the inner NR.
    const int    max_sweeps_triplet = 32;
    const double eps_triplet        = triplet_eps;  // per-call (default 1000)
    const double nr_tol_triplet     = 1e-7;
    const int    nr_max_iter_triplet = 30;
    const double blmin_triplet      = 1e-6;
    const double blmax_triplet      = 100.0;

    auto try_add = [&](int depth, int node,
                       const double* corr_clv, const double* corr_ls, int rb,
                       /* depth==0: side is ra_other, reused as its 5-BLO boundary
                        * CLV. Ignored when depth > 0. */
                       const double* side_clv_in = nullptr,
                       const double* side_ls_in  = nullptr,
                       /* depth==0: nb_other's CLV (from DFS seeding), the 5-BLO
                        * nb boundary. Ignored when depth > 0. */
                       const double* nb_clv_in = nullptr,
                       const double* nb_ls_in  = nullptr,
                       /* rb's CLV at (rb -> node), precomputed once per frame by
                        * the DFS body. Null fallback for test call sites. */
                       const double* rb_clv_in = nullptr,
                       const double* rb_ls_in  = nullptr) {
        PROF_SCOPE("try_add");
        // Per-candidate counters: ta.calls_* (candidates/roll), ta.depth_* mean
        // (avg regraft depth), ta.depth_dist_* histogram.
        COUNTER_INC(thorough ? "ta.calls_slow" : "ta.calls_fast");
        COUNTER_ADD(thorough ? "ta.depth_slow" : "ta.depth_fast", depth);
        COUNTER_HIST(thorough ? "ta.depth_dist_slow" : "ta.depth_dist_fast",
                     (uint64_t)depth);
        stk_nodes[depth] = node;
        std::memcpy(&stk_clvs[depth * KL],       corr_clv, KL * sizeof(double));
        std::memcpy(&stk_log_scales[depth * L_],  corr_ls,  L_ * sizeof(double));

        // Initial BLs (split/merge convention): B2 = B3 = bl[ra,rb]/2;
        // B1 = current bl[pu,pv].
        double B1 = bl_[prune_u * n_ + prune_v];   // pendant, unchanged by SPR
        double B2 = bl_[node * n_ + rb] * 0.5;     // pu—ra, half of merged
        double B3 = B2;                             // pu—rb, same length
        // 5-BLO extras (NaN unless 5-BLO ran at depth==0)
        double B4_ra_nb_other = std::numeric_limits<double>::quiet_NaN();
        double B5_ra_ra_other = std::numeric_limits<double>::quiet_NaN();

        // rb's CLV (rb -> node), used by both the 5-BLO and no-BLO paths. Use
        // the caller's precomputed pointer when present; else compute it here.
        const double* rb_clv_p;
        const double* rb_ls_p;
        if (rb_clv_in) {
            rb_clv_p = rb_clv_in;
            rb_ls_p  = rb_ls_in;
        } else {
            _clv(rb, node, rb_raw.data(), rb_ls.data());
            rb_clv_p = rb_raw.data();
            rb_ls_p  = rb_ls.data();
        }

        double total_ll;
        bool ran_triplet = false;

        // 5-BLO path at radius-1 + thorough; otherwise the no-BLO score path.
        bool use_5blo = (thorough && depth == 0);
        int  nb_other = -1, ra_other = -1;
        if (use_5blo) {
            nb_other = (node == nb1) ? nb2 : nb1;
            int ra_nbs[3];
            int ra_nc = tree.neighbors_of(node, ra_nbs);
            for (int i = 0; i < ra_nc; ++i)
                if (ra_nbs[i] != prune_u && ra_nbs[i] != rb)
                    { ra_other = ra_nbs[i]; break; }
            if (ra_other < 0) use_5blo = false;  // degenerate; fall through
        }

        if (use_5blo) {
            double bl5[5];
            // Pass pre-computed boundary CLVs to skip all 4 internal _clv()
            // calls inside 5-BLO (pv, rb, side, nb).
            total_ll = _optimize_five_branches_at_nni(
                prune_u, prune_v, node /*=ra*/, rb,
                nb_other, ra_other, bl5,
                pv_raw.data(), pv_ls.data(),
                rb_clv_p, rb_ls_p,
                side_clv_in,   side_ls_in,
                nb_clv_in,     nb_ls_in,
                fiveblo_eps);
            COUNTER_INC("5blo.applied");
            B2 = bl5[0];
            B1 = bl5[1];
            B3 = bl5[2];
            B4_ra_nb_other = bl5[3];
            B5_ra_ra_other = bl5[4];
            ran_triplet = true;

            model_->p_matrix(B1, P_B1.data());
            _p_apply(P_B1.data(), pv_raw.data(), pv_contrib_local.data());
            model_->p_matrix(B2, P_B2.data());
            _p_apply(P_B2.data(), corr_clv, contrib_ra.data());
            model_->p_matrix(B3, P_B3.data());
            _p_apply(P_B3.data(), rb_clv_p, contrib_rb.data());
            _clv_mul(contrib_ra.data(), contrib_rb.data(), pseudo_clv.data());
            for (int l = 0; l < L_; l++) pseudo_ls[l] = corr_ls[l] + rb_ls_p[l];
            _scale_clv_inplace(pseudo_clv.data(), pseudo_ls.data());
        } else {
            // No-BLO score path (FAST any depth; SLOW depth>0): score at the
            // convention BLs. FAST uses this directly; SLOW triplet BLO below
            // starts from it.
            model_->p_matrix(B2, P_half.data());
            _p_apply(P_half.data(), corr_clv, contrib_ra.data());
            _p_apply(P_half.data(), rb_clv_p, contrib_rb.data());

            _clv_mul(contrib_ra.data(), contrib_rb.data(), pseudo_clv.data());
            for (int l = 0; l < L_; l++) pseudo_ls[l] = corr_ls[l] + rb_ls_p[l];
            _scale_clv_inplace(pseudo_clv.data(), pseudo_ls.data());
            // FUSED_SCORE: one streaming reduction against wpv. ki-OUTER/l-INNER
            // so wpv/pseudo are read contiguously (l-outer thrashed cache on long
            // MSAs); l-inner FMA auto-vectorizes; ascending-ki accumulation
            // matches the else branch.
            double total_ll_noblo;
            const int* w = msa_->weights.data();
            if (fused_score) {
                // site_lik[l] = Σ_ki wpv[ki,l]·pseudo[ki,l], ki-outer + 4-wide FMA.
                double* sl = site_lik.data();
#ifdef __AVX2__
                int L4 = L_ & ~3;
                for (int l = 0; l < L4; l += 4) _mm256_storeu_pd(sl + l, _mm256_setzero_pd());
                for (int l = L4; l < L_; l++) sl[l] = 0.0;
                for (int ki = 0; ki < K_; ki++) {
                    const double* wr = wpv.data()        + (size_t)ki * L_;
                    const double* pr = pseudo_clv.data() + (size_t)ki * L_;
                    for (int l = 0; l < L4; l += 4) {
                        __m256d acc = _mm256_loadu_pd(sl + l);
                        acc = _mm256_fmadd_pd(_mm256_loadu_pd(wr + l),
                                              _mm256_loadu_pd(pr + l), acc);
                        _mm256_storeu_pd(sl + l, acc);
                    }
                    for (int l = L4; l < L_; l++) sl[l] += wr[l] * pr[l];
                }
#else
                for (int l = 0; l < L_; l++) sl[l] = 0.0;
                for (int ki = 0; ki < K_; ki++) {
                    const double* wr = wpv.data()        + (size_t)ki * L_;
                    const double* pr = pseudo_clv.data() + (size_t)ki * L_;
                    for (int l = 0; l < L_; l++) sl[l] += wr[l] * pr[l];
                }
#endif
                double acc = 0.0;
                for (int l = 0; l < L_; l++) {
                    double s = sl[l] < 1e-300 ? 1e-300 : sl[l];
                    acc += w[l] * (std::log(s) + pseudo_ls[l] + pv_ls[l]);
                }
                total_ll_noblo = acc;
            } else {
                _clv_mul(pseudo_clv.data(), pv_contrib.data(), final_clv.data());
                for (int l = 0; l < L_; l++) final_ls[l] = pseudo_ls[l] + pv_ls[l];
                _scale_clv_inplace(final_clv.data(), final_ls.data());
                for (int l = 0; l < L_; l++) {
                    double s = 0.0;
                    for (int ki = 0; ki < K_; ki++)
                        s += pi[ki] * final_clv[ki * L_ + l];
                    site_lik[l] = s < 1e-300 ? 1e-300 : s;
                }
                double acc = 0.0;
                for (int l = 0; l < L_; l++)
                    acc += w[l] * (std::log(site_lik[l]) + final_ls[l]);
                total_ll_noblo = acc;
            }
            total_ll = total_ll_noblo;

            if (thorough) {
            // Per-candidate copy of pv_contrib (since triplet BLO may update B1).
            std::memcpy(pv_contrib_local.data(), pv_contrib.data(), KL * sizeof(double));

            double prev_score = total_ll_noblo;

            [[maybe_unused]] int sweeps_run = 0;
            [[maybe_unused]] bool eps_break = false;
        for (int sweep = 0; sweep < max_sweeps_triplet; sweep++) {
            sweeps_run = sweep + 1;
            // ---- Optimize B1 (pendant pu—pv) ----
            //   A = pseudo_clv = contrib_ra * contrib_rb (combined at pu side)
            //   B = pv_raw     (CLV at pv, no pendant P)
            // pseudo_clv depends on contrib_ra & contrib_rb (latest), so build
            // it fresh each sweep — cheap.
            _clv_mul(contrib_ra.data(), contrib_rb.data(), pseudo_clv.data());
            if (_sumtable_enabled() && model_->eigenvalues()) {
                static thread_local std::vector<double> S_b1;
                if ((int)S_b1.size() < KL) S_b1.assign(KL, 0.0);
                _build_sumtable(pseudo_clv.data(), pv_raw.data(), S_b1.data());
                B1 = _nr_branch_with_sumtable(S_b1.data(),
                                              B1, blmin_triplet, blmax_triplet,
                                              nr_tol_triplet, nr_max_iter_triplet,
                                              nullptr);
                COUNTER_HIST("nr.iters.b1", g_last_nr_iters);
            } else {
                B1 = _nr_branch_with_clvs(pseudo_clv.data(), pv_raw.data(),
                                          B1, blmin_triplet, blmax_triplet,
                                          nr_tol_triplet, nr_max_iter_triplet);
                COUNTER_HIST("nr.iters.b1", g_last_nr_iters);
            }
            model_->p_matrix(B1, P_B1.data());
            _p_apply(P_B1.data(), pv_raw.data(), pv_contrib_local.data());

            // ---- Optimize B2 (pu—ra sub-edge) ----
            //   A = CLV at pu toward ra = pv_contrib_local * contrib_rb
            //   B = corr_clv
            _clv_mul(pv_contrib_local.data(), contrib_rb.data(), A_pu_ra.data());
            if (_sumtable_enabled() && model_->eigenvalues()) {
                static thread_local std::vector<double> S_b2;
                if ((int)S_b2.size() < KL) S_b2.assign(KL, 0.0);
                _build_sumtable(A_pu_ra.data(), corr_clv, S_b2.data());
                B2 = _nr_branch_with_sumtable(S_b2.data(),
                                              B2, blmin_triplet, blmax_triplet,
                                              nr_tol_triplet, nr_max_iter_triplet,
                                              nullptr);
                COUNTER_HIST("nr.iters.b2", g_last_nr_iters);
            } else {
                B2 = _nr_branch_with_clvs(A_pu_ra.data(), corr_clv,
                                          B2, blmin_triplet, blmax_triplet,
                                          nr_tol_triplet, nr_max_iter_triplet);
                COUNTER_HIST("nr.iters.b2", g_last_nr_iters);
            }
            model_->p_matrix(B2, P_B2.data());
            _p_apply(P_B2.data(), corr_clv, contrib_ra.data());

            // ---- Optimize B3 (pu—rb sub-edge) ----
            //   A = CLV at pu toward rb = pv_contrib_local * contrib_ra (latest)
            //   B = rb_raw
            _clv_mul(pv_contrib_local.data(), contrib_ra.data(), A_pu_rb.data());
            if (_sumtable_enabled() && model_->eigenvalues()) {
                static thread_local std::vector<double> S_b3;
                if ((int)S_b3.size() < KL) S_b3.assign(KL, 0.0);
                _build_sumtable(A_pu_rb.data(), rb_clv_p, S_b3.data());
                B3 = _nr_branch_with_sumtable(S_b3.data(),
                                              B3, blmin_triplet, blmax_triplet,
                                              nr_tol_triplet, nr_max_iter_triplet,
                                              nullptr);
                COUNTER_HIST("nr.iters.b3", g_last_nr_iters);
            } else {
                B3 = _nr_branch_with_clvs(A_pu_rb.data(), rb_clv_p,
                                          B3, blmin_triplet, blmax_triplet,
                                          nr_tol_triplet, nr_max_iter_triplet);
                COUNTER_HIST("nr.iters.b3", g_last_nr_iters);
            }
            model_->p_matrix(B3, P_B3.data());
            _p_apply(P_B3.data(), rb_clv_p, contrib_rb.data());

            // ---- Score at end of sweep ----
            _clv_mul(contrib_ra.data(), contrib_rb.data(), pseudo_clv.data());
            for (int l = 0; l < L_; l++) pseudo_ls[l] = corr_ls[l] + rb_ls_p[l];
            _scale_clv_inplace(pseudo_clv.data(), pseudo_ls.data());

            _clv_mul(pseudo_clv.data(), pv_contrib_local.data(), final_clv.data());
            for (int l = 0; l < L_; l++) final_ls[l] = pseudo_ls[l] + pv_ls[l];
            _scale_clv_inplace(final_clv.data(), final_ls.data());

            for (int l = 0; l < L_; l++) {
                double s = 0.0;
                for (int ki = 0; ki < K_; ki++)
                    s += pi[ki] * final_clv[ki * L_ + l];
                site_lik[l] = s < 1e-300 ? 1e-300 : s;
            }
            total_ll = 0.0;
            const int* w = msa_->weights.data();
            for (int l = 0; l < L_; l++)
                total_ll += w[l] * (std::log(site_lik[l]) + final_ls[l]);

            if (total_ll - prev_score < eps_triplet) { eps_break = true; break; }
            prev_score = total_ll;
        }
            COUNTER_HIST("triplet.sweeps", sweeps_run);
            COUNTER_INC(eps_break ? "triplet.break.eps" : "triplet.break.maxsweeps");
            ran_triplet = true;
            }   // end if (thorough)  -- inner triplet conditional
        }   // end else  -- closes the no-BLO score path

        if (static_cast<int>(heap.size()) < k || total_ll > heap[0].score) {
            int d = depth + 1;
            LikelihoodSPRCandidate cand;
            cand.log_likelihood = total_ll;
            cand.prune_u = prune_u; cand.prune_v = prune_v;
            cand.ra = node;         cand.rb = rb;
            cand.nb1 = nb1;         cand.nb2 = nb2;
            cand.path_nodes.assign(stk_nodes.begin(), stk_nodes.begin() + d);
            // path_nodes is the only path data commit() consumes (LOG_COMMITS>=3
            // + local-BLO enumeration); no directional CLVs are stored.

            // Store optimized BLs: B1/B2/B3 (3 overlap edges, both paths);
            // B4/B5 only from the 5-BLO depth==0 path.
            if (ran_triplet) {
                cand.triplet_bl_pendant = B1;
                cand.triplet_bl_ra      = B2;
                cand.triplet_bl_rb      = B3;
                if (!std::isnan(B4_ra_nb_other))
                    cand.bl_ra_nb_other = B4_ra_nb_other;
                if (!std::isnan(B5_ra_ra_other))
                    cand.bl_ra_ra_other = B5_ra_ra_other;
            }

            HeapEntry entry{total_ll, ctr++, std::move(cand)};
            if (static_cast<int>(heap.size()) < k) {
                heap.push_back(std::move(entry));
                std::push_heap(heap.begin(), heap.end());
            } else {
                std::pop_heap(heap.begin(), heap.end());
                heap.back() = std::move(entry);
                std::push_heap(heap.begin(), heap.end());
            }
        }
    };

    // DFS frames index a thread_local slot pool instead of owning their
    // came_from CLV/ls. A popped slot is freed at the END of the iteration,
    // not before — pushes during the iteration must not clobber the slot
    // still being read.
    struct DFSFrame {
        int node, came_from, depth, buf_idx;
    };
    static thread_local std::vector<std::vector<double>> dfs_clv_pool;
    static thread_local std::vector<std::vector<double>> dfs_ls_pool;
    static thread_local std::vector<int> dfs_free_list;

    // Free-list starts with all existing slots; alloc grows the pool on demand.
    dfs_free_list.clear();
    for (int i = (int)dfs_clv_pool.size() - 1; i >= 0; --i)
        dfs_free_list.push_back(i);

    auto alloc_dfs_slot = [&]() -> int {
        int idx;
        if (!dfs_free_list.empty()) {
            idx = dfs_free_list.back();
            dfs_free_list.pop_back();
        } else {
            idx = (int)dfs_clv_pool.size();
            dfs_clv_pool.emplace_back(KL, 0.0);
            dfs_ls_pool.emplace_back(L_, 0.0);
        }
        if ((int)dfs_clv_pool[idx].size() < KL) dfs_clv_pool[idx].assign(KL, 0.0);
        if ((int)dfs_ls_pool [idx].size() < L_) dfs_ls_pool [idx].assign(L_, 0.0);
        return idx;
    };
    auto free_dfs_slot = [&](int idx) {
        dfs_free_list.push_back(idx);
    };

    std::vector<DFSFrame> dfs_stack;
    dfs_stack.reserve(tree.n_nodes);

    // Seed both directions from prune_u via P_merged. nb_seed_clv[i] holds
    // _clv(sibling_i, prune_u), kept (not overwritten) so the depth-0 5-BLO can
    // reuse it as the nb_other boundary: frame.node==ra uses
    // nb_seed_clv[(ra == nb1) ? 0 : 1].
    static thread_local std::vector<double> nb_seed_clv[2], nb_seed_ls[2];
    resize_if_smaller(nb_seed_clv[0], KL);
    resize_if_smaller(nb_seed_clv[1], KL);
    resize_if_smaller(nb_seed_ls [0], L_);
    resize_if_smaller(nb_seed_ls [1], L_);
    for (int i = 0; i < 2; i++) {
        int first   = (i == 0) ? nb1 : nb2;
        int sibling = (i == 0) ? nb2 : nb1;
        int idx = alloc_dfs_slot();
        _clv(sibling, prune_u, nb_seed_clv[i].data(), nb_seed_ls[i].data());
        _p_apply(P_merged.data(), nb_seed_clv[i].data(), dfs_clv_pool[idx].data());
        // P-apply doesn't change scale: came_from_ls = sib_ls
        std::memcpy(dfs_ls_pool[idx].data(), nb_seed_ls[i].data(),
                    L_ * sizeof(double));
        dfs_stack.push_back({first, prune_u, 0, idx});
    }

    // DFS body scratch — overwritten every DFS step, reused across all candidates.
    static thread_local std::vector<double> side_raw, side_ls, P_node_side, side_contrib,
                                            corr_clv, corr_ls,
                                            P_rb_node, next_came_from;
    resize_if_smaller(side_raw, KL);
    resize_if_smaller(side_ls,  L_);
    resize_if_smaller(P_node_side, KK);
    resize_if_smaller(side_contrib, KL);
    resize_if_smaller(corr_clv, KL);
    resize_if_smaller(corr_ls,  L_);
    resize_if_smaller(P_rb_node, KK);
    resize_if_smaller(next_came_from, KL);

    while (!dfs_stack.empty()) {
        DFSFrame frame = dfs_stack.back();
        dfs_stack.pop_back();
        // frame.buf_idx still occupied — read from it, then free at end of iter.
        const double* came_from_clv_p = dfs_clv_pool[frame.buf_idx].data();
        const double* came_from_ls_p  = dfs_ls_pool [frame.buf_idx].data();

        // FRAME-level NNI dedup at depth==0: if (pu, ra) was seen this round,
        // skip ONLY the radius-1 scoring; the deeper DFS push still happens.
        // Keys are NOT direction-canonicalized — (X,Y) and (Y,X) start 5-BLO
        // from different split/merge BLs and land at slightly different LLs, so
        // saved_cap can pick the better; canonicalizing would forfeit that.
        // only_ra set => single-target re-score: leave seen_radius1_ untouched
        // (side-effect-free, target never dedup-skipped).
        bool frame_is_duplicate = false;
        if (frame.depth == 0 && only_ra == INVALID) {
            int64_t key = (int64_t)prune_u * (int64_t)n_ + (int64_t)frame.node;
            // Flat-bitset check-and-set: seen this round iff entry == gen_.
            if (seen_radius1_[key] == seen_radius1_gen_) {
                COUNTER_INC("dedup.frame_skip");
                frame_is_duplicate = true;
            } else {
                seen_radius1_[key] = seen_radius1_gen_;
            }
        }

        int outward[2];
        int nc = tree.neighbors_except(frame.node, frame.came_from, outward);

        // Precompute both child CLVs once per frame: each child is used as
        // both `side` and (in try_add) `rb` across the two iterations, so
        // computing here avoids a double _clv per child.
        static thread_local std::vector<double> child_clv[2], child_ls[2];
        resize_if_smaller(child_clv[0], KL);
        resize_if_smaller(child_clv[1], KL);
        resize_if_smaller(child_ls [0], L_);
        resize_if_smaller(child_ls [1], L_);
        for (int j = 0; j < nc; j++)
            _clv(outward[j], frame.node,
                 child_clv[j].data(), child_ls[j].data());

        for (int i = 0; i < nc; i++) {
            int rb   = outward[i];
            int side = outward[1 - i];

            // side and rb both bind into the precomputed 2-slot buffer; rb_*_p
            // is threaded through to try_add.
            const double* side_clv_p = child_clv[1 - i].data();
            const double* side_ls_p  = child_ls [1 - i].data();
            const double* rb_clv_p   = child_clv[i].data();
            const double* rb_ls_p    = child_ls [i].data();

            // Direct p_arr read: (frame.node, side) is always a live tree edge,
            // so the slot is populated; skips _p_for_edge's memcpy.
            const double* P_node_side_ptr =
                &p_arr_[(frame.node * 3 + _slot_of(frame.node, side)) * KK];
            _p_apply(P_node_side_ptr, side_clv_p, side_contrib.data());
            // side_ls unchanged: P-apply doesn't affect scale

            _clv_mul(came_from_clv_p, side_contrib.data(), corr_clv.data());
            for (int l = 0; l < L_; l++) corr_ls[l] = came_from_ls_p[l] + side_ls_p[l];
            _scale_clv_inplace(corr_clv.data(), corr_ls.data());

            // Score only within the [radius_min, radius] window (depth+1 = prune-
            // to-regraft distance), and skip when the depth==0 frame was a dup.
            if (!frame_is_duplicate && frame.depth + 1 >= radius_min
                && (only_ra == INVALID
                    || (frame.node == only_ra && rb == only_rb))) {
                // depth==0: frame.node is nb1 or nb2; pass the matching nb_other
                // seed CLV. (Ignored by try_add at depth > 0.)
                const double* nb_clv_p = nullptr;
                const double* nb_ls_p  = nullptr;
                if (frame.depth == 0) {
                    int nb_buf_idx = (frame.node == nb1) ? 0 : 1;
                    nb_clv_p = nb_seed_clv[nb_buf_idx].data();
                    nb_ls_p  = nb_seed_ls [nb_buf_idx].data();
                }
                try_add(frame.depth, frame.node, corr_clv.data(), corr_ls.data(), rb,
                        side_clv_p, side_ls_p,
                        nb_clv_p, nb_ls_p,
                        rb_clv_p, rb_ls_p);
            }

            if (frame.depth + 1 < radius && !tree.is_leaf(rb)) {
                // (rb, frame.node) is always a tree edge — direct cache read.
                const double* P_rb_node_ptr =
                    &p_arr_[(rb * 3 + _slot_of(rb, frame.node)) * KK];
                int new_idx = alloc_dfs_slot();
                // Write straight into the pool slot (skips the next_came_from
                // intermediate + its memcpy).
                _p_apply(P_rb_node_ptr, corr_clv.data(),
                         dfs_clv_pool[new_idx].data());
                // P-apply doesn't change scale: came_from_ls = corr_ls
                std::memcpy(dfs_ls_pool[new_idx].data(),
                            corr_ls.data(),        L_ * sizeof(double));
                dfs_stack.push_back({rb, frame.node, frame.depth + 1, new_idx});
            }
        }

        // Done reading the popped frame's buffer; release it for reuse.
        free_dfs_slot(frame.buf_idx);
    }

    // Sort best first (descending log-likelihood)
    std::sort(heap.begin(), heap.end(),
              [](const HeapEntry& a, const HeapEntry& b) {
                  return a.score > b.score;
              });

    std::vector<LikelihoodSPRCandidate> result;
    result.reserve(heap.size());
    for (auto& e : heap)
        result.push_back(std::move(e.cand));
    return result;
}