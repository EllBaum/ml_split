// likelihood_scorer_blo.cpp
// Part of the single-TU umbrella (likelihood_scorer_unit.cpp); not compiled standalone.
#include "likelihood_scorer.h"

// _nr_branch_with_clvs -- pure NR helper, no global state.
// Operates on supplied flanking CLVs A and B (no this-edge P applied), used by
// triplet BLO in try_add, which keeps A/B in scratch and never mutates the tree.
// f/fp are the 1st/2nd derivs of the log-likelihood; NR maintains a bracket
// [xl,xh] on the sign of f, caps the step at blmax/max_iter, and on
// anti-curvature (df<=0) steps -f with magnitude |f|/|df|. Converges on
// |f|<tol or |dx|<tol; no line search.
double LikelihoodScorer::_nr_branch_with_clvs(const double* A, const double* B,
                                              double t_init,
                                              double blmin, double blmax,
                                              double tol, int max_iter) const {
    return _nr_branch_with_clvs_with_status(A, B, t_init, blmin, blmax,
                                            tol, max_iter, /*converged=*/nullptr);
}

double LikelihoodScorer::_nr_branch_with_clvs_with_status(
    const double* A, const double* B,
    double t_init,
    double blmin, double blmax,
    double tol, int max_iter,
    bool* converged) const {
    PROF_SCOPE("_nr_branch_with_clvs");
    int KL = K_ * L_;
    int KK = K_ * K_;
    const double* pi = model_->pi();
    const int*    w  = msa_->weights.data();
    const double  dxmax = blmax / max_iter;

    if (converged) *converged = false;

    // PB/dPB/ddPB = P/dP/ddP applied to B (not yet times A or summed over k).
    std::vector<double> P(KK), dP(KK), ddP(KK);
    std::vector<double> PB(KL), dPB(KL), ddPB(KL);

    double xl = blmin;
    double xh = blmax;
    double t  = std::max(blmin, std::min(blmax, t_init));

    int iters_run = 0;
    for (int iter = 0; iter < max_iter; iter++) {
        iters_run = iter + 1;
        model_->p_matrix(t, P.data());
        model_->dp_ddp_matrix(t, dP.data(), ddP.data());

        _p_apply(P.data(),   B, PB.data());
        _p_apply(dP.data(),  B, dPB.data());
        _p_apply(ddP.data(), B, ddPB.data());

        // lik[l] = sum_k pi[k] * PB[k,l] * A[k,l]  (and d/dd analogues).
        double f = 0.0, fp = 0.0;
        for (int l = 0; l < L_; l++) {
            double s = 0.0, ds = 0.0, dds = 0.0;
            for (int k = 0; k < K_; k++) {
                double a = A[k*L_+l];
                s   += pi[k] * PB[k*L_+l]   * a;
                ds  += pi[k] * dPB[k*L_+l]  * a;
                dds += pi[k] * ddPB[k*L_+l] * a;
            }
            double sl = s > 1e-300 ? s : 1e-300;
            double r  = ds  / sl;
            double r2 = dds / sl;
            f  += w[l] * r;
            fp += w[l] * (r2 - r*r);
        }
        // Sign-flip (minimizes neg-log-lik).
        double f_rx  = -f;
        double df_rx = -fp;

        if (!std::isfinite(f_rx) || !std::isfinite(df_rx)) break;

        double dx;
        if (df_rx > 0.0) {
            if (std::fabs(f_rx) < tol) {
                if (converged) *converged = true;
                break;
            }
            if (f_rx < 0.0) xl = t;
            else            xh = t;
            dx = -f_rx / df_rx;
        } else {
            // Anti-curvature: step -f with magnitude |f|/|df|.
            dx = -f_rx / std::fabs(df_rx);
        }

        if (dx >  dxmax) dx =  dxmax;
        if (dx < -dxmax) dx = -dxmax;

        // Clip to bracket.
        if (t + dx < xl) dx = xl - t;
        if (t + dx > xh) dx = xh - t;

        if (std::fabs(dx) < tol) {
            if (converged) *converged = true;
            break;
        }

        t += dx;
        if (t < blmin) t = blmin;
        if (t > blmax) t = blmax;
    }

    g_last_nr_iters = iters_run;
    return t;
}

// _build_sumtable
// State-major S[m*L + l] (matches _p_apply output, no transpose needed):
//   S[m,l] = (R @ B)[m,l] * (Lpi @ A)[m,l],  Lpi[m,k] = L_mat[k,m]*pi[k].
// Built with one fused _p_apply_pair (both projections in registers, no
// intermediate buffer). Paired with the state-major reader in
// _nr_branch_with_sumtable.
void LikelihoodScorer::_build_sumtable(const double* A, const double* B,
                                       double* S_out) const {
    PROF_SCOPE("_build_sumtable");
    const double* eig_R = model_->R_mat();
    const double* eig_L = model_->L_mat();
    const double* pi    = model_->pi();
    assert(eig_R && eig_L);

    // Lpi[m,k] = L_mat[k,m] * pi[k], built once (constant for scorer lifetime).
    if (!Lpi_mat_built_) {
        Lpi_mat_.assign(K_ * K_, 0.0);
        for (int m = 0; m < K_; m++)
            for (int k = 0; k < K_; k++)
                Lpi_mat_[m * K_ + k] = eig_L[k * K_ + m] * pi[k];
        Lpi_mat_built_ = true;
    }

    _p_apply_pair(Lpi_mat_.data(), A, eig_R, B, S_out);
}

// _nr_branch_with_sumtable

// AVX-512 state-major K=20 NR inner kernel.
#if defined(__AVX512F__) || defined(__AVX2__)
__attribute__((target("avx512f,avx512vl,avx512dq")))
static void _nr_inner_k20_sm_avx512(const double* S, int L,
                                     const double* __restrict__ e,
                                     const double* __restrict__ de,
                                     const double* __restrict__ dde,
                                     const int*    __restrict__ w,
                                     double& f_out, double& fp_out)
{
    const int Lblk = L - (L % 8);
    __m512d v_f  = _mm512_setzero_pd();
    __m512d v_fp = _mm512_setzero_pd();
    const __m512d eps_v = _mm512_set1_pd(1e-300);
    const __m512d one_v = _mm512_set1_pd(1.0);

    for (int l0 = 0; l0 < Lblk; l0 += 8) {
        __m512d s_v   = _mm512_setzero_pd();
        __m512d ds_v  = _mm512_setzero_pd();
        __m512d dds_v = _mm512_setzero_pd();
        for (int m = 0; m < 20; m++) {
            __m512d s_m = _mm512_loadu_pd(S + m * L + l0);
            s_v   = _mm512_fmadd_pd(_mm512_set1_pd(e  [m]), s_m, s_v);
            ds_v  = _mm512_fmadd_pd(_mm512_set1_pd(de [m]), s_m, ds_v);
            dds_v = _mm512_fmadd_pd(_mm512_set1_pd(dde[m]), s_m, dds_v);
        }
        s_v = _mm512_max_pd(s_v, eps_v);
        __m512d inv_v = _mm512_div_pd(one_v, s_v);
        __m512d dlog  = _mm512_mul_pd(ds_v, inv_v);

        __m256i w_i = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + l0));
        __m512d w_v = _mm512_cvtepi32_pd(w_i);

        v_f = _mm512_fmadd_pd(w_v, dlog, v_f);
        __m512d term = _mm512_sub_pd(_mm512_mul_pd(dds_v, inv_v),
                                     _mm512_mul_pd(dlog,  dlog));
        v_fp = _mm512_fmadd_pd(w_v, term, v_fp);
    }
    f_out  = _mm512_reduce_add_pd(v_f);
    fp_out = _mm512_reduce_add_pd(v_fp);

    // Scalar tail.
    for (int l = Lblk; l < L; l++) {
        double s_l = 0.0, ds_l = 0.0, dds_l = 0.0;
        for (int m = 0; m < 20; m++) {
            double sml = S[m * L + l];
            s_l   += e  [m] * sml;
            ds_l  += de [m] * sml;
            dds_l += dde[m] * sml;
        }
        if (s_l < 1e-300) s_l = 1e-300;
        double inv  = 1.0 / s_l;
        double dlog = ds_l * inv;
        f_out  += w[l] * dlog;
        fp_out += w[l] * (dds_l * inv - dlog * dlog);
    }
}
#endif

double LikelihoodScorer::_nr_branch_with_sumtable(
    const double* S,
    double t_init,
    double blmin, double blmax,
    double tol, int max_iter,
    bool* converged) const
{
    PROF_SCOPE("_nr_branch_with_sumtable");
    const double* eigvals = model_->eigenvalues();
    const int*    w       = msa_->weights.data();
    const double  dxmax   = blmax / max_iter;
    const int K = K_;
    const int L = L_;

    if (converged) *converged = false;

    double t  = t_init;
    if (t < blmin) t = blmin;
    if (t > blmax) t = blmax;
    double xl = blmin, xh = blmax;

    assert(K <= 20);
    alignas(32) double e[20], de[20], dde[20];

    int iters_run = 0;
    for (int it = 0; it < max_iter; it++) {
        iters_run = it + 1;
        for (int m = 0; m < K; m++) {
            double v  = std::exp(eigvals[m] * t);
            e  [m] = v;
            de [m] = eigvals[m] * v;
            dde[m] = eigvals[m] * eigvals[m] * v;
        }

        double f = 0.0, fp = 0.0;

#if defined(__AVX2__) && defined(__FMA__)
        if (K == 20 && g_have_avx512) {
            COUNTER_INC("nr.k20.avx512");
            _nr_inner_k20_sm_avx512(S, L, e, de, dde, w, f, fp);
        } else if (K == 20) {
            COUNTER_INC("nr.k20.avx2");
            // State-major S[m*L+l]: vectorize over L (4 patterns/__m256d), m
            // inner with e/de/dde broadcasts. Stride-1 contiguous S loads that
            // the prefetcher streams cleanly.
            int Lblk = L - (L % 4);
            __m256d v_f  = _mm256_setzero_pd();
            __m256d v_fp = _mm256_setzero_pd();

            for (int l0 = 0; l0 < Lblk; l0 += 4) {
                __m256d s_v   = _mm256_setzero_pd();
                __m256d ds_v  = _mm256_setzero_pd();
                __m256d dds_v = _mm256_setzero_pd();
                for (int m = 0; m < 20; m++) {
                    __m256d s_m = _mm256_loadu_pd(S + m * L + l0);
                    s_v   = _mm256_fmadd_pd(_mm256_set1_pd(e  [m]), s_m, s_v);
                    ds_v  = _mm256_fmadd_pd(_mm256_set1_pd(de [m]), s_m, ds_v);
                    dds_v = _mm256_fmadd_pd(_mm256_set1_pd(dde[m]), s_m, dds_v);
                }
                __m256d eps   = _mm256_set1_pd(1e-300);
                s_v           = _mm256_max_pd(s_v, eps);
                __m256d inv_v = _mm256_div_pd(_mm256_set1_pd(1.0), s_v);
                __m256d dlog  = _mm256_mul_pd(ds_v, inv_v);

                __m128i w_i = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + l0));
                __m256d w_v = _mm256_cvtepi32_pd(w_i);

                v_f = _mm256_fmadd_pd(w_v, dlog, v_f);
                __m256d term = _mm256_sub_pd(_mm256_mul_pd(dds_v, inv_v),
                                             _mm256_mul_pd(dlog,  dlog));
                v_fp = _mm256_fmadd_pd(w_v, term, v_fp);
            }
            double fR[4], fpR[4];
            _mm256_storeu_pd(fR,  v_f);
            _mm256_storeu_pd(fpR, v_fp);
            f  += fR [0] + fR [1] + fR [2] + fR [3];
            fp += fpR[0] + fpR[1] + fpR[2] + fpR[3];

            for (int l = Lblk; l < L; l++) {
                double s_l = 0.0, ds_l = 0.0, dds_l = 0.0;
                for (int m = 0; m < 20; m++) {
                    double sml = S[m * L + l];
                    s_l   += e  [m] * sml;
                    ds_l  += de [m] * sml;
                    dds_l += dde[m] * sml;
                }
                if (s_l < 1e-300) s_l = 1e-300;
                double inv  = 1.0 / s_l;
                double dlog = ds_l * inv;
                f  += w[l] * dlog;
                fp += w[l] * (dds_l * inv - dlog * dlog);
            }
        } else
#endif
        {
            // Scalar fallback for general K (e.g. K=4 DNA). State-major S.
            for (int l = 0; l < L; l++) {
                double s = 0.0, ds = 0.0, dds = 0.0;
                for (int m = 0; m < K; m++) {
                    double sml = S[m * L + l];
                    s   += e  [m] * sml;
                    ds  += de [m] * sml;
                    dds += dde[m] * sml;
                }
                if (s < 1e-300) s = 1e-300;
                double inv = 1.0 / s;
                double dlog = ds * inv;
                f  += w[l] * dlog;
                fp += w[l] * (dds * inv - dlog * dlog);
            }
        }

        double f_rx  = -f;
        double df_rx = -fp;

        if (!std::isfinite(f_rx) || !std::isfinite(df_rx)) break;

        if (df_rx > 0.0) {
            if (std::fabs(f_rx) < tol) {
                if (converged) *converged = true;
                break;
            }
            if (f_rx < 0.0) xl = t; else xh = t;
        }

        double dx;
        if (df_rx > 0.0) dx = -f_rx / df_rx;
        else             dx = -f_rx / std::fabs(df_rx);

        if (dx >  dxmax) dx =  dxmax;
        if (dx < -dxmax) dx = -dxmax;

        if (t + dx < xl) dx = xl - t;
        if (t + dx > xh) dx = xh - t;

        if (std::fabs(dx) < tol) {
            if (converged) *converged = true;
            break;
        }

        t += dx;
        if (t < blmin) t = blmin;
        if (t > blmax) t = blmax;
    }

    g_last_nr_iters = iters_run;
    return t;
}

// Thin wrapper around the sumtable/CLV NR. On non-convergence, keep the
// original BL (treat "hit max_iter without |f|<tol or |dx|<tol" as failure).
double LikelihoodScorer::_optimize_single_branch(int u, int v,
                                                   double blmin, double blmax,
                                                   double tol, int max_iter) {
    PROF_SCOPE("_optimize_single_branch");
    double t_orig = bl_[u * n_ + v];
    if (t_orig <= 0.0) return t_orig;

    int KL = K_ * L_;
    std::vector<double> A(KL), B(KL), lsA(L_), lsB(L_);
    _clv(u, v, A.data(), lsA.data());
    _clv(v, u, B.data(), lsB.data());

    bool converged = false;
    double t_new;

    if (_sumtable_enabled() && model_->eigenvalues()) {
        static thread_local std::vector<double> S_out;
        if ((int)S_out.size() < KL) S_out.assign(KL, 0.0);
        _build_sumtable(A.data(), B.data(), S_out.data());
        t_new = _nr_branch_with_sumtable(
            S_out.data(), t_orig, blmin, blmax, tol, max_iter, &converged);

#ifdef SUMTABLE_BLO_VERIFY
        bool old_conv = false;
        double t_old = _nr_branch_with_clvs_with_status(
            A.data(), B.data(), t_orig, blmin, blmax, tol, max_iter, &old_conv);
        double rel = std::fabs(t_old - t_new) / std::max(1e-12, std::fabs(t_old));
        if (rel > 1e-9)
            std::fprintf(stderr,
                "SUMTABLE_BLO_VERIFY single (%d,%d) old=%.15g new=%.15g rel=%.3g\n",
                u, v, t_old, t_new, rel);
#endif
    } else {
        t_new = _nr_branch_with_clvs_with_status(
            A.data(), B.data(), t_orig, blmin, blmax, tol, max_iter, &converged);
    }

    double t = converged ? t_new : t_orig;
    bl_[u * n_ + v] = t;
    bl_[v * n_ + u] = t;
    return t;
}

// optimize_branch_lengths

double LikelihoodScorer::optimize_branch_lengths(const std::string& mode,
                                                   double lh_epsilon,
                                                   int    max_sweeps,
                                                   int    n_sweeps,
                                                   double blmin,
                                                   double blmax,
                                                   const LikelihoodSPRCandidate* cand) {
    PROF_SCOPE("optimize_branch_lengths");
    const Tree& tree = *tree_;

    if (mode == "triplet")
        return _optimize_triplets(lh_epsilon, max_sweeps, blmin, blmax);

    // mode="lbfgsb" (TDL): joint L-BFGS-B. Falls back to "thorough".
    std::string eff_mode = (mode == "lbfgsb") ? "thorough" : mode;

    std::vector<std::pair<int,int>> edges;

    if (mode == "local") {
        // Edges adjacent to the recent SPR move; one fast sweep if cand==null.
        if (!cand) {
            return optimize_branch_lengths("fast", lh_epsilon, 1, 1, blmin, blmax);
        }
        int pu = cand->prune_u, pv = cand->prune_v;
        int ra = cand->ra,      rb = cand->rb;
        int nb1= cand->nb1,     nb2= cand->nb2;

        std::set<std::pair<int,int>> local_set;
        auto add_edge = [&](int a, int b) {
            if (a > b) std::swap(a, b);
            local_set.insert({a, b});
        };
        add_edge(pu, pv); add_edge(nb1, nb2);
        add_edge(pu, ra); add_edge(pu, rb);
        for (int nid : cand->path_nodes) {
            int tw = toward_[nid];
            if (tw != INVALID) add_edge(nid, tw);
        }
        for (auto [a, b] : local_set)
            if (bl_[a * n_ + b] > 0.0) edges.push_back({a, b});
        eff_mode = "fast";
        n_sweeps = 1;
    } else {
        std::vector<int> order = tree.postorder(calc_start_, 0);
        std::set<std::pair<int,int>> seen;
        for (int v : order) {
            int u = toward_[v];
            if (u == INVALID) continue;
            auto e = (u < v) ? std::make_pair(u,v) : std::make_pair(v,u);
            if (!seen.count(e)) { edges.push_back({v,u}); seen.insert(e); }
        }
        int u0 = 0, v0 = calc_start_;
        auto e0 = (u0 < v0) ? std::make_pair(u0,v0) : std::make_pair(v0,u0);
        if (!seen.count(e0)) edges.push_back({u0, v0});
    }

    // local mode runs n_sweeps (=1); fast/thorough run max_sweeps. Not in use now.
    int sweeps_to_do = (mode == "local")
                       ? n_sweeps
                       : (eff_mode == "fast" || eff_mode == "thorough")
                         ? max_sweeps
                         : n_sweeps;

    double prev_score = score_;

    for (int s = 0; s < sweeps_to_do; s++) {
        for (auto [u, v] : edges) {
            if (bl_[u * n_ + v] > 0.0) {
                double t_before = bl_[u * n_ + v];
                double t_after  = _optimize_single_branch(u, v, blmin, blmax);
                // No-move skip: _optimize_single_branch writes only bl_, so an
                // unchanged branch leaves scorer state byte-identical and the
                // refresh/dirty/propagate below would be pure waste. Exact ==
                // is deliberate (bit-for-bit same value).
                if (t_after == t_before) continue;
                if (eff_mode == "thorough") {
                    _full_recompute();
                } else {
                    // Refresh p_arr for this edge so later branches in the sweep
                    // see fresh P. p_matrix(t) is identical for both directions
                    // -- compute once, memcpy to the mirror slot.
                    double t_uv = bl_[u * n_ + v];
                    int s_uv = _slot_of(u, v);
                    int s_vu = _slot_of(v, u);
                    if (s_uv >= 0) {
                        model_->p_matrix(t_uv, &p_arr_[(u * 3 + s_uv) * K_ * K_]);
                        if (s_vu >= 0)
                            std::memcpy(&p_arr_[(v * 3 + s_vu) * K_ * K_],
                                        &p_arr_[(u * 3 + s_uv) * K_ * K_],
                                        K_ * K_ * sizeof(double));
                    } else if (s_vu >= 0) {
                        model_->p_matrix(t_uv, &p_arr_[(v * 3 + s_vu) * K_ * K_]);
                    }

                    // The p_arr update invalidates any CLV that traversed (u,v):
                    // dirty all slots, then re-propagate the toward slots up from
                    // u and v to calc_start. Cheaper than precise bookkeeping.
                    std::fill(dirty_.begin(), dirty_.end(), 1);
                    std::unordered_set<int> visited_blo;
                    _propagate_up_ll(u, visited_blo);
                    _propagate_up_ll(v, visited_blo);
                }
            }
        }
        if (eff_mode != "thorough") _full_recompute();

        double improvement = score_ - prev_score;
        if ((eff_mode == "fast" || eff_mode == "thorough") && improvement < lh_epsilon)
            break;
        prev_score = score_;
    }

    return score_;
}

// _optimize_triplets

double LikelihoodScorer::_optimize_triplets(double lh_epsilon,
                                             int    max_sweeps,
                                             double blmin,
                                             double blmax,
                                             double tol,
                                             int    max_iter) {
    const Tree& tree = *tree_;
    const double* pi      = model_->pi();
    const int*    w       = msa_->weights.data();
    int KL = K_ * L_;

    // 1D NR on one arm, given Q = product of the other two arms' P-applied CLVs.
    // R = raw CLV snapshot of this arm (states-major).
    auto nr_1d = [&](const std::vector<double>& R,
                     const std::vector<double>& Q,
                     double t_init) -> double {
        double t = t_init;

        auto site_ll = [&](double tv) -> double {
            std::vector<double> P(K_*K_);
            model_->p_matrix(tv, P.data());
            double ll = 0.0;
            for (int l = 0; l < L_; l++) {
                double s = 0.0;
                for (int k = 0; k < K_; k++) {
                    // (R @ P.T)[k,l] states-major: sum_j R[j*L+l] * P[k*K+j]
                    double rpt = 0.0;
                    for (int j = 0; j < K_; j++) rpt += R[j*L_+l] * P[k*K_+j];
                    s += pi[k] * rpt * Q[k*L_+l];
                }
                if (s < 1e-300) s = 1e-300;
                ll += w[l] * std::log(s);
            }
            return ll;
        };

        double ll_cur = site_ll(t);

        for (int iter = 0; iter < max_iter; iter++) {
            std::vector<double> P(K_*K_), dP(K_*K_), ddP(K_*K_);
            model_->p_matrix(t, P.data());
            model_->dp_ddp_matrix(t, dP.data(), ddP.data());

            std::vector<double> lik(L_,0), dlik(L_,0), ddlik(L_,0);
            for (int l = 0; l < L_; l++) {
                for (int k = 0; k < K_; k++) {
                    double rp=0, rdp=0, rddp=0;
                    for (int j = 0; j < K_; j++) {
                        rp   += R[j*L_+l] * P  [k*K_+j];
                        rdp  += R[j*L_+l] * dP [k*K_+j];
                        rddp += R[j*L_+l] * ddP[k*K_+j];
                    }
                    lik  [l] += pi[k] * rp   * Q[k*L_+l];
                    dlik [l] += pi[k] * rdp  * Q[k*L_+l];
                    ddlik[l] += pi[k] * rddp * Q[k*L_+l];
                }
                if (lik[l] < 1e-300) lik[l] = 1e-300;
            }

            double f=0, fp=0;
            for (int l = 0; l < L_; l++) {
                double r  = dlik [l] / lik[l];
                double r2 = ddlik[l] / lik[l];
                f  += w[l] * r;
                fp += w[l] * (r2 - r*r);
            }

            if (std::fabs(f) < tol) break;
            double step = (fp < 0.0) ? f / fp : -f * 0.01;
            double t_new = std::max(blmin, std::min(blmax, t - step));

            double ll_new = site_ll(t_new);
            for (int ls = 0; ls < 10 && ll_new < ll_cur - 1e-10; ls++) {
                step  *= 0.5;
                t_new  = std::max(blmin, std::min(blmax, t - step));
                ll_new = site_ll(t_new);
            }
            if (std::fabs(t_new - t) < 1e-10) { t = t_new; break; }
            t = t_new; ll_cur = ll_new;
        }
        return t;
    };

    // P-apply R (K,L) through P (K,K): out[k,l] = sum_j R[j,l] * P[k,j].
    auto p_apply = [&](const std::vector<double>& P,
                       const std::vector<double>& R,
                       std::vector<double>& out) {
        for (int k = 0; k < K_; k++)
            for (int l = 0; l < L_; l++) {
                double s = 0.0;
                for (int j = 0; j < K_; j++) s += R[j*L_+l] * P[k*K_+j];
                out[k*L_+l] = s;
            }
    };

    double prev_score = score_;

    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        std::vector<int> order = tree.postorder(calc_start_, 0);

        for (int v : order) {
            if (tree.is_leaf(v)) continue;
            int nbs[3]; int nc = tree.neighbors_of(v, nbs);
            if (nc != 3) continue;

            int a = nbs[0], b = nbs[1], c = nbs[2];
            double ta = bl_[v*n_+a], tb = bl_[v*n_+b], tc = bl_[v*n_+c];
            if (ta <= 0.0 || tb <= 0.0 || tc <= 0.0) continue;

            double ll_before = score_;

            std::vector<double> Ra(KL), Rb(KL), Rc(KL);
            std::vector<double> lsA(L_), lsB(L_), lsC(L_);
            _clv(a, v, Ra.data(), lsA.data());
            _clv(b, v, Rb.data(), lsB.data());
            _clv(c, v, Rc.data(), lsC.data());

            std::vector<double> Pb(K_*K_), Pc(K_*K_), Pa(K_*K_);
            std::vector<double> RbP(KL), RcP(KL), RaP(KL);

            // Step 1: optimize ta -- Q_bc = (Rb@P(tb)) * (Rc@P(tc))
            model_->p_matrix(tb, Pb.data()); model_->p_matrix(tc, Pc.data());
            p_apply(Pb, Rb, RbP); p_apply(Pc, Rc, RcP);
            std::vector<double> Q_bc(KL);
            for (int i = 0; i < KL; i++) Q_bc[i] = RbP[i] * RcP[i];
            double ta_new = nr_1d(Ra, Q_bc, ta);

            // Step 2: optimize tb -- Q_ac = (Ra@P(ta_new)) * (Rc@P(tc))
            model_->p_matrix(ta_new, Pa.data());
            p_apply(Pa, Ra, RaP);
            std::vector<double> Q_ac(KL);
            for (int i = 0; i < KL; i++) Q_ac[i] = RaP[i] * RcP[i];
            double tb_new = nr_1d(Rb, Q_ac, tb);

            // Step 3: optimize tc -- Q_ab = (Ra@P(ta_new)) * (Rb@P(tb_new))
            model_->p_matrix(tb_new, Pb.data());
            p_apply(Pb, Rb, RbP);
            std::vector<double> Q_ab(KL);
            for (int i = 0; i < KL; i++) Q_ab[i] = RaP[i] * RbP[i];
            double tc_new = nr_1d(Rc, Q_ab, tc);

            bl_[v*n_+a] = bl_[a*n_+v] = ta_new;
            bl_[v*n_+b] = bl_[b*n_+v] = tb_new;
            bl_[v*n_+c] = bl_[c*n_+v] = tc_new;
            _full_recompute();

            // Restore if the global LL decreased.
            if (score_ < ll_before - 1e-10) {
                bl_[v*n_+a] = bl_[a*n_+v] = ta;
                bl_[v*n_+b] = bl_[b*n_+v] = tb;
                bl_[v*n_+c] = bl_[c*n_+v] = tc;
                _full_recompute();
            }
        }

        double improvement = score_ - prev_score;
        if (improvement < lh_epsilon) break;
        prev_score = score_;
    }

    return score_;
}



std::vector<double> LikelihoodScorer::get_bl_array() const {
    const Tree& tree = *tree_;
    std::vector<double> bl(n_ * 3, 0.0);
    for (int u = 0; u < n_; u++) {
        int base = u * 3;
        for (int k = 0; k < 3; k++) {
            int v = tree.neighbors[base + k];
            if (v != INVALID) {
                bl[base + k] = bl_[u * n_ + v];
            }
        }
    }
    return bl;
}

// _optimize_five_branches_at_nni
//
// Gauss-Seidel BLO over the 5 branches around the post-SPR NNI center edge
// (pu, ra), for an SPR(pu, pv, ra, rb) with ra a pre-SPR neighbor of pu
// (radius-1 / NNI). Post-SPR neighborhood:
//
//          pv ------- pu ------- ra ------- ra_other
//                     |           |
//                     rb          nb_other
//
//   (pu, ra)       new NNI center edge
//   (pu, pv)       preserved
//   (pu, rb)       new (rb was ra's pre-SPR neighbor)
//   (ra, nb_other) new (the old bypass edge nb_other-pu, now spliced)
//   (ra, ra_other) preserved
//
// The 4 boundary subtrees (pv, rb, nb_other, ra_other) don't change shape, so
// their pre-SPR _clv CLVs are valid boundary conditions for the post-SPR
// mini-Felsenstein.
//
// Initial BLs (split/merge convention): (pu,ra)/(pu,pv)/(ra,ra_other)
// preserved; (pu,rb) = bl_pre(ra,rb)/2 (split); (ra,nb_other) =
// bl_pre(pu,ra) + bl_pre(pu,nb_other) (merged bypass sum).
//
// Caller must verify ra in {nb1,nb2} and supply nb_other / ra_other. Returns
// the optimized post-SPR LL; bl_out[5] = {(pu,ra),(pu,pv),(pu,rb),
// (ra,nb_other),(ra,ra_other)}.
double LikelihoodScorer::_optimize_five_branches_at_nni(
    int prune_u, int prune_v, int ra, int rb,
    int nb_other, int ra_other,
    double bl_out[5],
    const double* pv_clv_in, const double* pv_ls_in,
    const double* rb_clv_in, const double* rb_ls_in,
    const double* ro_clv_in, const double* ro_ls_in,
    const double* nb_clv_in, const double* nb_ls_in,
    double eps_5blo_in, const double* init_bl5) const
{
    PROF_SCOPE("_5blo_nni");
    COUNTER_INC("5blo.calls");

    const int    KL = K_ * L_;
    const int    KK = K_ * K_;
    const double* pi = model_->pi();
    const int*    w  = msa_->weights.data();

    // -- 1. Boundary CLVs (4) -------------------------------------------------
    // Computed against the pre-SPR tree; the subtrees beyond pv/rb/nb_other/
    // ra_other don't change in the SPR, so they're valid post-SPR too. When the
    // caller passes {pv,rb,ro,nb}_clv_in (it already has them -- pv_raw, rb_raw,
    // side_raw, and the DFS seed buffer), reuse them; else _clv fresh.
    thread_local std::vector<double> bnd_pv_clv_buf, bnd_pv_ls_buf;
    thread_local std::vector<double> bnd_rb_clv_buf, bnd_rb_ls_buf;
    thread_local std::vector<double> bnd_nb_clv_buf, bnd_nb_ls_buf;
    thread_local std::vector<double> bnd_ro_clv_buf, bnd_ro_ls_buf;

    const double* bnd_pv_clv;
    const double* bnd_pv_ls;
    if (pv_clv_in) {
        COUNTER_INC("5blo.bnd_pv.reused");
        bnd_pv_clv = pv_clv_in;
        bnd_pv_ls  = pv_ls_in;
    } else {
        COUNTER_INC("5blo.bnd_pv.computed");
        bnd_pv_clv_buf.resize(KL); bnd_pv_ls_buf.resize(L_);
        _clv(prune_v, prune_u, bnd_pv_clv_buf.data(), bnd_pv_ls_buf.data());
        bnd_pv_clv = bnd_pv_clv_buf.data();
        bnd_pv_ls  = bnd_pv_ls_buf.data();
    }

    const double* bnd_rb_clv;
    const double* bnd_rb_ls;
    if (rb_clv_in) {
        COUNTER_INC("5blo.bnd_rb.reused");
        bnd_rb_clv = rb_clv_in;
        bnd_rb_ls  = rb_ls_in;
    } else {
        COUNTER_INC("5blo.bnd_rb.computed");
        bnd_rb_clv_buf.resize(KL); bnd_rb_ls_buf.resize(L_);
        _clv(rb, ra, bnd_rb_clv_buf.data(), bnd_rb_ls_buf.data());
        bnd_rb_clv = bnd_rb_clv_buf.data();
        bnd_rb_ls  = bnd_rb_ls_buf.data();
    }

    const double* bnd_nb_clv;
    const double* bnd_nb_ls;
    if (nb_clv_in) {
        COUNTER_INC("5blo.bnd_nb.reused");
        bnd_nb_clv = nb_clv_in;
        bnd_nb_ls  = nb_ls_in;
    } else {
        COUNTER_INC("5blo.bnd_nb.computed");
        bnd_nb_clv_buf.resize(KL); bnd_nb_ls_buf.resize(L_);
        _clv(nb_other, prune_u, bnd_nb_clv_buf.data(), bnd_nb_ls_buf.data());
        bnd_nb_clv = bnd_nb_clv_buf.data();
        bnd_nb_ls  = bnd_nb_ls_buf.data();
    }

    const double* bnd_ro_clv;
    const double* bnd_ro_ls;
    if (ro_clv_in) {
        COUNTER_INC("5blo.bnd_ro.reused");
        bnd_ro_clv = ro_clv_in;
        bnd_ro_ls  = ro_ls_in;
    } else {
        COUNTER_INC("5blo.bnd_ro.computed");
        bnd_ro_clv_buf.resize(KL); bnd_ro_ls_buf.resize(L_);
        _clv(ra_other, ra, bnd_ro_clv_buf.data(), bnd_ro_ls_buf.data());
        bnd_ro_clv = bnd_ro_clv_buf.data();
        bnd_ro_ls  = bnd_ro_ls_buf.data();
    }

    // -- 2. Initial branch lengths (split/merge convention) -------------------
    // Merge hook: init_bl5 supplies them directly (order = bl_out), so the
    // hypothetical connector nodes need no entry in bl_.
    double bl_pu_ra, bl_pu_pv, bl_pu_rb, bl_ra_nb, bl_ra_ro;
    if (init_bl5) {
        bl_pu_ra = init_bl5[0];
        bl_pu_pv = init_bl5[1];
        bl_pu_rb = init_bl5[2];
        bl_ra_nb = init_bl5[3];
        bl_ra_ro = init_bl5[4];
    } else {
        bl_pu_ra = bl_[prune_u * n_ + ra];
        bl_pu_pv = bl_[prune_u * n_ + prune_v];
        bl_pu_rb = 0.5 * bl_[ra * n_ + rb];                          // split
        bl_ra_nb = bl_[prune_u * n_ + ra] + bl_[prune_u * n_ + nb_other]; // merged
        bl_ra_ro = bl_[ra * n_ + ra_other];
    }

    // -- 3. Directional CLVs at pu and ra (3 each) ----------------------------
    // clv_pu_toward_X = CLV at pu from its two sides other than X (and likewise
    // for ra). Materialized up front; the 2 affected ones refreshed after each
    // branch update. Each is a few _p_apply + _clv_mul of boundary contribs.
    thread_local std::vector<double> P_pu_pv(KK), P_pu_rb(KK), P_pu_ra(KK);
    thread_local std::vector<double> P_ra_nb(KK), P_ra_ro(KK);
    thread_local std::vector<double> contrib_pv(KL), contrib_rb(KL),
                                     contrib_nb(KL), contrib_ro(KL);
    thread_local std::vector<double> clv_pu_toward_ra(KL),
                                     clv_pu_toward_pv(KL),
                                     clv_pu_toward_rb(KL),
                                     clv_ra_toward_pu(KL),
                                     clv_ra_toward_nb(KL),
                                     clv_ra_toward_ro(KL);
    thread_local std::vector<double> ls_pu_toward_ra(L_),
                                     ls_pu_toward_pv(L_),
                                     ls_pu_toward_rb(L_),
                                     ls_ra_toward_pu(L_),
                                     ls_ra_toward_nb(L_),
                                     ls_ra_toward_ro(L_);
    // These thread_local buffers are constructor-sized on the FIRST call in a
    // thread only; that size is never re-evaluated. A later call in the same
    // thread with a larger K_*L_ (e.g. a different alignment with more patterns)
    // would otherwise reuse the first call's smaller buffers and let the SIMD
    // kernels write past their ends (heap-buffer-overflow). Resize to the
    // current dims every call -- a no-op when unchanged -- matching the
    // bnd_*_clv_buf.resize() pattern above.
    for (auto* v : {&P_pu_pv, &P_pu_rb, &P_pu_ra, &P_ra_nb, &P_ra_ro})
        v->resize(KK);
    for (auto* v : {&contrib_pv, &contrib_rb, &contrib_nb, &contrib_ro,
                    &clv_pu_toward_ra, &clv_pu_toward_pv, &clv_pu_toward_rb,
                    &clv_ra_toward_pu, &clv_ra_toward_nb, &clv_ra_toward_ro})
        v->resize(KL);
    for (auto* v : {&ls_pu_toward_ra, &ls_pu_toward_pv, &ls_pu_toward_rb,
                    &ls_ra_toward_pu, &ls_ra_toward_nb, &ls_ra_toward_ro})
        v->resize(L_);

    // Boundary contribs through their immediate edges; used by all three
    // "toward" combinations at each internal node.
    auto refresh_boundary_contribs = [&]() {
        model_->p_matrix(bl_pu_pv, P_pu_pv.data());
        _p_apply(P_pu_pv.data(), bnd_pv_clv, contrib_pv.data());

        model_->p_matrix(bl_pu_rb, P_pu_rb.data());
        _p_apply(P_pu_rb.data(), bnd_rb_clv, contrib_rb.data());

        model_->p_matrix(bl_ra_nb, P_ra_nb.data());
        _p_apply(P_ra_nb.data(), bnd_nb_clv, contrib_nb.data());

        model_->p_matrix(bl_ra_ro, P_ra_ro.data());
        _p_apply(P_ra_ro.data(), bnd_ro_clv, contrib_ro.data());
    };

    auto combine = [&](const double* a_clv, const double* a_ls,
                       const double* b_clv, const double* b_ls,
                       double* out_clv, double* out_ls) {
        _clv_mul(a_clv, b_clv, out_clv);
        for (int l = 0; l < L_; ++l) out_ls[l] = a_ls[l] + b_ls[l];
        _scale_clv_inplace(out_clv, out_ls);
    };

    // Seed toward_ra / toward_pu from boundary contribs only; the cross-
    // directional CLVs (which need the opposite internal node) are built after,
    // and the whole thing relaxes Gauss-Seidel-style across sweeps.
    auto rebuild_pu_directionals = [&]() {
        combine(contrib_pv.data(), bnd_pv_ls,
                contrib_rb.data(), bnd_rb_ls,
                clv_pu_toward_ra.data(), ls_pu_toward_ra.data());
    };
    auto rebuild_ra_directionals = [&]() {
        combine(contrib_nb.data(), bnd_nb_ls,
                contrib_ro.data(), bnd_ro_ls,
                clv_ra_toward_pu.data(), ls_ra_toward_pu.data());
    };

    refresh_boundary_contribs();
    rebuild_pu_directionals();
    rebuild_ra_directionals();

    // Cross-directional CLVs: the "through (pu,ra)" contribution is p_matrix(
    // bl_pu_ra) applied to the opposite internal's toward-us CLV.
    thread_local std::vector<double> P_pu_ra_mat(KK);
    thread_local std::vector<double> pu_thru_ra_at_ra(KL),
                                     ra_thru_pu_at_pu(KL);
    P_pu_ra_mat.resize(KK);            // see thread_local-resize note above
    pu_thru_ra_at_ra.resize(KL);
    ra_thru_pu_at_pu.resize(KL);

    auto rebuild_internal_pairs = [&]() {
        model_->p_matrix(bl_pu_ra, P_pu_ra_mat.data());
        _p_apply(P_pu_ra_mat.data(), clv_pu_toward_ra.data(),
                 pu_thru_ra_at_ra.data());
        _p_apply(P_pu_ra_mat.data(), clv_ra_toward_pu.data(),
                 ra_thru_pu_at_pu.data());

        combine(ra_thru_pu_at_pu.data(), ls_ra_toward_pu.data(),
                contrib_rb.data(),       bnd_rb_ls,
                clv_pu_toward_pv.data(), ls_pu_toward_pv.data());
        combine(ra_thru_pu_at_pu.data(), ls_ra_toward_pu.data(),
                contrib_pv.data(),       bnd_pv_ls,
                clv_pu_toward_rb.data(), ls_pu_toward_rb.data());

        combine(pu_thru_ra_at_ra.data(), ls_pu_toward_ra.data(),
                contrib_ro.data(),       bnd_ro_ls,
                clv_ra_toward_nb.data(), ls_ra_toward_nb.data());
        combine(pu_thru_ra_at_ra.data(), ls_pu_toward_ra.data(),
                contrib_nb.data(),       bnd_nb_ls,
                clv_ra_toward_ro.data(), ls_ra_toward_ro.data());
    };

    rebuild_internal_pairs();

    // -- 4. Gauss-Seidel sweeps over the 5 branches ---------------------------
    // Each branch: NR via sumtable from the directional CLVs at its endpoints,
    // then refresh the affected boundary-contrib + internal-pair CLVs.
    thread_local std::vector<double> S(KL);  // sumtable
    S.resize(KL);                            // see thread_local-resize note above

    const double nr_tol_5blo      = 1e-7;  // match production triplet
    const int    nr_max_iter_5blo = 30;

    auto blo_branch = [&](double& bl_inout,
                          const double* clv_a,
                          const double* clv_b,
                          double bl_min, double bl_max) -> double {
        _build_sumtable(clv_a, clv_b, S.data());
        return _nr_branch_with_sumtable(S.data(),
                                        bl_inout, bl_min, bl_max,
                                        nr_tol_5blo, nr_max_iter_5blo,
                                        nullptr);
    };

    const int    max_sweeps_5blo = 32;
    const double eps_5blo        = eps_5blo_in;  // per-call (default 1000)
    const double bl_min          = 1e-6;
    const double bl_max          = 100.0;

    double prev_score;
    {
        // Initial LL (same formula as the in-loop check), integrated at (pu,ra).
        thread_local std::vector<double> site_lik_init;
        if ((int)site_lik_init.size() < L_) site_lik_init.assign(L_, 0.0);
        for (int l = 0; l < L_; ++l) {
            double s = 0.0;
            for (int k = 0; k < K_; ++k) {
                s += pi[k]
                   * clv_pu_toward_ra[k * L_ + l]
                   * ra_thru_pu_at_pu[k * L_ + l];
            }
            site_lik_init[l] = (s < 1e-300 ? 1e-300 : s);
        }
        prev_score = 0.0;
        for (int l = 0; l < L_; ++l) {
            prev_score += w[l] * (std::log(site_lik_init[l])
                                + ls_pu_toward_ra[l] + ls_ra_toward_pu[l]);
        }
    }
    double total_ll = prev_score;

    for (int sweep = 0; sweep < max_sweeps_5blo; ++sweep) {

        // Branch 1: (pu, ra). Reads clv_pu_toward_ra, clv_ra_toward_pu.
        bl_pu_ra = blo_branch(bl_pu_ra,
                              clv_pu_toward_ra.data(),
                              clv_ra_toward_pu.data(),
                              bl_min, bl_max);
        model_->p_matrix(bl_pu_ra, P_pu_ra_mat.data());
        _p_apply(P_pu_ra_mat.data(), clv_ra_toward_pu.data(),
                 ra_thru_pu_at_pu.data());
        combine(ra_thru_pu_at_pu.data(), ls_ra_toward_pu.data(),
                contrib_rb.data(),       bnd_rb_ls,
                clv_pu_toward_pv.data(), ls_pu_toward_pv.data());

        // Branch 2: (pu, pv). Reads clv_pu_toward_pv, bnd_pv_clv.
        bl_pu_pv = blo_branch(bl_pu_pv,
                              clv_pu_toward_pv.data(),
                              bnd_pv_clv,
                              bl_min, bl_max);
        model_->p_matrix(bl_pu_pv, P_pu_pv.data());
        _p_apply(P_pu_pv.data(), bnd_pv_clv, contrib_pv.data());
        combine(ra_thru_pu_at_pu.data(), ls_ra_toward_pu.data(),
                contrib_pv.data(),       bnd_pv_ls,
                clv_pu_toward_rb.data(), ls_pu_toward_rb.data());

        // Branch 3: (pu, rb). Reads clv_pu_toward_rb, bnd_rb_clv.
        bl_pu_rb = blo_branch(bl_pu_rb,
                              clv_pu_toward_rb.data(),
                              bnd_rb_clv,
                              bl_min, bl_max);
        model_->p_matrix(bl_pu_rb, P_pu_rb.data());
        _p_apply(P_pu_rb.data(), bnd_rb_clv, contrib_rb.data());
        combine(contrib_pv.data(), bnd_pv_ls,
                contrib_rb.data(), bnd_rb_ls,
                clv_pu_toward_ra.data(), ls_pu_toward_ra.data());
        _p_apply(P_pu_ra_mat.data(), clv_pu_toward_ra.data(),
                 pu_thru_ra_at_ra.data());
        combine(pu_thru_ra_at_ra.data(), ls_pu_toward_ra.data(),
                contrib_ro.data(),       bnd_ro_ls,
                clv_ra_toward_nb.data(), ls_ra_toward_nb.data());

        // Branch 4: (ra, nb_other). Reads clv_ra_toward_nb, bnd_nb_clv.
        bl_ra_nb = blo_branch(bl_ra_nb,
                              clv_ra_toward_nb.data(),
                              bnd_nb_clv,
                              bl_min, bl_max);
        model_->p_matrix(bl_ra_nb, P_ra_nb.data());
        _p_apply(P_ra_nb.data(), bnd_nb_clv, contrib_nb.data());
        combine(pu_thru_ra_at_ra.data(), ls_pu_toward_ra.data(),
                contrib_nb.data(),       bnd_nb_ls,
                clv_ra_toward_ro.data(), ls_ra_toward_ro.data());

        // Branch 5: (ra, ra_other). Reads clv_ra_toward_ro, bnd_ro_clv.
        bl_ra_ro = blo_branch(bl_ra_ro,
                              clv_ra_toward_ro.data(),
                              bnd_ro_clv,
                              bl_min, bl_max);
        model_->p_matrix(bl_ra_ro, P_ra_ro.data());
        _p_apply(P_ra_ro.data(), bnd_ro_clv, contrib_ro.data());
        combine(contrib_nb.data(), bnd_nb_ls,
                contrib_ro.data(), bnd_ro_ls,
                clv_ra_toward_pu.data(), ls_ra_toward_pu.data());
        _p_apply(P_pu_ra_mat.data(), clv_ra_toward_pu.data(),
                 ra_thru_pu_at_pu.data());

        // LL at (pu,ra): sum_k pi[k]*clv_pu_toward_ra[k]*ra_thru_pu_at_pu[k].
        thread_local std::vector<double> site_lik(L_);
        site_lik.resize(L_);                 // see thread_local-resize note above
        for (int l = 0; l < L_; ++l) {
            double s = 0.0;
            for (int k = 0; k < K_; ++k) {
                s += pi[k]
                   * clv_pu_toward_ra[k * L_ + l]
                   * ra_thru_pu_at_pu[k * L_ + l];
            }
            site_lik[l] = (s < 1e-300 ? 1e-300 : s);
        }
        total_ll = 0.0;
        for (int l = 0; l < L_; ++l) {
            total_ll += w[l] * (std::log(site_lik[l])
                              + ls_pu_toward_ra[l] + ls_ra_toward_pu[l]);
        }

        if (total_ll - prev_score < eps_5blo) {
            COUNTER_INC("5blo.eps_break");
            break;
        }
        prev_score = total_ll;
    }

    bl_out[0] = bl_pu_ra;
    bl_out[1] = bl_pu_pv;
    bl_out[2] = bl_pu_rb;
    bl_out[3] = bl_ra_nb;
    bl_out[4] = bl_ra_ro;

    return total_ll;
}