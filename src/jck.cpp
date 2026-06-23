// jck.cpp - K-state Jukes-Cantor substitution model (JC69 / Poisson).

#include "jck.h"
#include "gtr.h"        // GTR::eigen (shared symmetric eigensolver)
#include "counters.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <algorithm>

// ================================================================== //
// JC69 — now JCK alias, see JCModel section below

// ================================================================== //
// JCK — K-state Jukes-Cantor (JC69=JCK(4), JCModel=JCK alias)                                     //
// ================================================================== //

JCK::JCK(int K) : K_(K), pi_(K, 1.0 / K) { _setup_eigen(); }

// Analytic eigendecomposition of the JC rate matrix. JC = equal exchange rates
// + uniform pi
void JCK::_setup_eigen() {
    const int K = K_;
    assert(K <= 20);   // fixed-size eig storage (max K = 20, JCAA)

    // All temporaries on the stack (no heap allocations) — matches JTT/GTR, so
    // the model contributes no allocations that would shift the heap layout.
    double Q[400] = {};
    const double off = 1.0 / (K - 1.0);
    for (int i = 0; i < K; i++)
        for (int j = 0; j < K; j++)
            Q[i * K + j] = (i == j) ? -1.0 : off;

    // Symmetrize S = diag(sqrt_pi) Q diag(inv_sqrt_pi). pi is uniform so S == Q,
    // but follow GTR/JTT's construction verbatim for convention parity.
    double sqrt_pi[20], inv_sqrt_pi[20];
    for (int i = 0; i < K; i++) {
        sqrt_pi[i]     = std::sqrt(pi_[i]);
        inv_sqrt_pi[i] = 1.0 / sqrt_pi[i];
    }
    double S[400];
    for (int i = 0; i < K; i++)
        for (int j = 0; j < K; j++)
            S[i * K + j] = sqrt_pi[i] * Q[i * K + j] * inv_sqrt_pi[j];

    // Symmetric eigendecomposition (reuse GTR's tred2+tqli solver).
    double d[20], U[400];
    GTR::symmetric_eigen(S, d, U, K);
    for (int i = 0; i < K; i++) eigenvalues_[i] = d[i];

    // Same L/R formulas as JTT/GTR: L = diag(inv_sqrt_pi) U, R = U^T diag(sqrt_pi).
    for (int i = 0; i < K; i++)
        for (int j = 0; j < K; j++) {
            L_mat_[i * K + j] = inv_sqrt_pi[i] * U[i * K + j];
            R_mat_[i * K + j] = U[j * K + i] * sqrt_pi[j];
        }
}

void JCK::p_matrix(double t, double* out) const {
    const int KK = K_ * K_;
    if (t <= 0.0) {
        for (int i = 0; i < KK; i++) out[i] = 0.0;
        for (int i = 0; i < K_; i++) out[i * K_ + i] = 1.0;
        return;
    }
    auto it = cache_.find(t);
    if (it != cache_.end()) {
        COUNTER_INC("p_matrix.jc.hit");
        std::memcpy(out, it->second.data(), KK * sizeof(double));
        return;
    }
    COUNTER_INC("p_matrix.jc.miss");
    // e = exp(-K*t/(K-1))
    double e = std::exp(-static_cast<double>(K_) * t / (K_ - 1.0));
    double p = (1.0 + (K_ - 1.0) * e) / K_;  // diagonal
    double q = (1.0 - e)               / K_;  // off-diagonal
    for (int i = 0; i < K_; i++)
        for (int j = 0; j < K_; j++)
            out[i * K_ + j] = (i == j) ? p : q;
    cache_[t] = std::vector<double>(out, out + KK);
}

void JCK::dp_ddp_matrix(double t, double* dp, double* ddp) const {
    if (t <= 0.0) t = 1e-10;
    double Kd  = static_cast<double>(K_);
    double Km1 = Kd - 1.0;
    double e   = std::exp(-Kd * t / Km1);
    double d_diag   = -e;
    double d_off    =  e / Km1;
    double dd_diag  =  Kd / Km1 * e;
    double dd_off   = -Kd / (Km1 * Km1) * e;
    for (int i = 0; i < K_; i++)
        for (int j = 0; j < K_; j++) {
            if (i == j) { dp[i*K_+j] = d_diag;  ddp[i*K_+j] = dd_diag; }
            else         { dp[i*K_+j] = d_off;   ddp[i*K_+j] = dd_off;  }
        }
}
