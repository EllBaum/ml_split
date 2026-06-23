// empirical_aa_model.cpp - empirical amino-acid substitution model base.

#include "empirical_aa_model.h"
#include "counters.h"
#include "prof.h"
#include "gtr.h"  // GTR::symmetric_eigen reused for eigendecomposition
#include <cmath>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <algorithm>

// ================================================================== //
// EmpiricalAAModel                                                    //
// ================================================================== //

EmpiricalAAModel::EmpiricalAAModel(const double exch[190],
                                   const double freqs_in[20])
{
    // Normalize frequencies
    double sum = 0.0;
    for (int i = 0; i < K_; i++) sum += freqs_in[i];
    for (int i = 0; i < K_; i++) pi_[i] = freqs_in[i] / sum;

    // Build Q: Q[i,j] = exch[i,j] * pi[j]  for i!=j
    double Q[K_*K_] = {};
    int idx = 0;
    for (int i = 0; i < K_; i++)
        for (int j = i+1; j < K_; j++) {
            Q[i*K_+j] = exch[idx] * pi_[j];
            Q[j*K_+i] = exch[idx] * pi_[i];
            ++idx;
        }
    // Diagonal
    for (int i = 0; i < K_; i++) {
        double row = 0.0;
        for (int j = 0; j < K_; j++) if (j!=i) row += Q[i*K_+j];
        Q[i*K_+i] = -row;
    }
    // Normalise: mean rate = 1
    double scale = 0.0;
    for (int i = 0; i < K_; i++) scale += pi_[i] * (-Q[i*K_+i]);
    if (scale > 0.0)
        for (int i = 0; i < K_*K_; i++) Q[i] /= scale;

    // Symmetrise: S = diag(sqrt_pi) @ Q @ diag(inv_sqrt_pi)
    double sqrt_pi[K_], inv_sqrt_pi[K_];
    for (int i = 0; i < K_; i++) {
        sqrt_pi[i]     = std::sqrt(pi_[i]);
        inv_sqrt_pi[i] = 1.0 / sqrt_pi[i];
    }
    double S[K_*K_];
    for (int i = 0; i < K_; i++)
        for (int j = 0; j < K_; j++)
            S[i*K_+j] = sqrt_pi[i] * Q[i*K_+j] * inv_sqrt_pi[j];

    // TODO: eliminate near-zero-frequency states (raxml `eliminate_zero_states`
    // in eigen.c) before eigendecomposition. Not needed for JTT (all pi > 0.014)
    // but required for safety with other empirical models or fitted frequencies
    // that may push pi near zero. Approach: pull out rows/cols with pi <= 1e-7,
    // eigendecompose the reduced matrix, pad eigenvalues to 0 and eigenvectors
    // to identity for eliminated states.
    double d[K_], U[K_*K_];
    GTR::symmetric_eigen(S, d, U, K_);
    for (int i = 0; i < K_; i++) eigenvalues_[i] = d[i];

    // L_mat[i][j] = inv_sqrt_pi[i] * U[i][j]
    // R_mat[i][j] = U[j][i] * sqrt_pi[j]
    for (int i = 0; i < K_; i++)
        for (int j = 0; j < K_; j++) {
            L_mat_[i*K_+j] = inv_sqrt_pi[i] * U[i*K_+j];
            R_mat_[i*K_+j] = U[j*K_+i]      * sqrt_pi[j];
        }
}

void EmpiricalAAModel::p_matrix(double t, double* out) const {
    PROF_SCOPE("p_matrix");
    const int KK = K_ * K_;
    if (t <= 0.0) {
        for (int i = 0; i < KK; i++) out[i] = 0.0;
        for (int i = 0; i < K_; i++) out[i*K_+i] = 1.0;
        return;
    }
    auto it = cache_.find(t);
    if (it != cache_.end()) {
        COUNTER_INC("p_matrix.aa.hit");
        std::memcpy(out, it->second.data(), KK * sizeof(double));
        return;
    }
    COUNTER_INC("p_matrix.aa.miss");
    double exp_d[K_];
    for (int j = 0; j < K_; j++) exp_d[j] = std::exp(eigenvalues_[j] * t);

    double tmp[K_*K_];
    for (int i = 0; i < K_; i++)
        for (int j = 0; j < K_; j++)
            tmp[i*K_+j] = L_mat_[i*K_+j] * exp_d[j];

    for (int i = 0; i < K_; i++)
        for (int k = 0; k < K_; k++) {
            double s = 0.0;
            for (int j = 0; j < K_; j++) s += tmp[i*K_+j] * R_mat_[j*K_+k];
            out[i*K_+k] = s;
        }
    cache_[t] = std::vector<double>(out, out + KK);
}

void EmpiricalAAModel::dp_ddp_matrix(double t, double* dp, double* ddp) const {
    if (t <= 0.0) t = 1e-10;
    double exp_d[K_];
    for (int j = 0; j < K_; j++) exp_d[j] = std::exp(eigenvalues_[j] * t);

    double tmp1[K_*K_], tmp2[K_*K_];
    for (int i = 0; i < K_; i++)
        for (int j = 0; j < K_; j++) {
            tmp1[i*K_+j] = L_mat_[i*K_+j] * eigenvalues_[j]                  * exp_d[j];
            tmp2[i*K_+j] = L_mat_[i*K_+j] * eigenvalues_[j]*eigenvalues_[j]  * exp_d[j];
        }
    for (int i = 0; i < K_; i++)
        for (int k = 0; k < K_; k++) {
            double s1 = 0.0, s2 = 0.0;
            for (int j = 0; j < K_; j++) {
                s1 += tmp1[i*K_+j] * R_mat_[j*K_+k];
                s2 += tmp2[i*K_+j] * R_mat_[j*K_+k];
            }
            dp[i*K_+k]  = s1;
            ddp[i*K_+k] = s2;
        }
}
