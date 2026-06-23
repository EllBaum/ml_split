// gtr.cpp - General Time-Reversible model + shared symmetric eigensolver.

#include "gtr.h"
#include "counters.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <algorithm>


// GTR — symmetric eigendecomposition                                  //
// ================================================================== //
//
// Real-symmetric eigendecomposition for the symmetrized rate matrix S.
// Uses the textbook two-stage approach:
//   1. Householder tridiagonalization (K-2 reflections, O(K^3) work).
//      A = Q T Q^T where T is symmetric tridiagonal.
//   2. Implicit QL with Wilkinson shifts on T (O(K^2) per eigenvalue).
//      T = V D V^T where D is diagonal.
//   Combine: A = (Q V) D (Q V)^T.
//
// This replaces an earlier Jacobi implementation that did not converge
// for K=20. Machine-precision eigenvectors at all sizes we use (K=4 GTR,
// K=20 empirical AA models).
//
// On entry: A[K*K] = symmetric matrix (row-major), modified in place.
// On exit:  d[K]   = eigenvalues
//           V[K*K] = eigenvectors in columns (row-major)
//                    so A V[:,k] = d[k] V[:,k] for each k.

void GTR::symmetric_eigen(double* A, double* d, double* V, int K) {
    // === Stage 1: Householder tridiagonalization (Numerical Recipes "tred2") ===
    //
    // Reduces A to symmetric tridiagonal T with diagonal d[] and sub/superdiagonal e[].
    // Accumulates orthogonal Q in V such that V^T A V = T.
    std::vector<double> e(K, 0.0);
    for (int i = 0; i < K * K; i++) V[i] = A[i];   // copy A → V (working)

    for (int i = K - 1; i >= 1; i--) {
        int    l = i - 1;
        double h = 0.0;
        double scale = 0.0;
        if (l > 0) {
            for (int k = 0; k <= l; k++) scale += std::fabs(V[i*K + k]);
            if (scale == 0.0) {
                // Skip transformation
                e[i] = V[i*K + l];
            } else {
                for (int k = 0; k <= l; k++) {
                    V[i*K + k] /= scale;
                    h += V[i*K + k] * V[i*K + k];
                }
                double f = V[i*K + l];
                double g = (f >= 0.0) ? -std::sqrt(h) : std::sqrt(h);
                e[i] = scale * g;
                h -= f * g;
                V[i*K + l] = f - g;
                f = 0.0;
                for (int j = 0; j <= l; j++) {
                    V[j*K + i] = V[i*K + j] / h;     // store the eigenvector accumulator
                    g = 0.0;
                    for (int k = 0; k <= j; k++)        g += V[j*K + k] * V[i*K + k];
                    for (int k = j + 1; k <= l; k++)    g += V[k*K + j] * V[i*K + k];
                    e[j] = g / h;
                    f   += e[j] * V[i*K + j];
                }
                double hh = f / (h + h);
                for (int j = 0; j <= l; j++) {
                    f    = V[i*K + j];
                    g    = e[j] - hh * f;
                    e[j] = g;
                    for (int k = 0; k <= j; k++)
                        V[j*K + k] -= (f * e[k] + g * V[i*K + k]);
                }
            }
        } else {
            e[i] = V[i*K + l];
        }
        d[i] = h;
    }

    d[0] = 0.0;
    e[0] = 0.0;

    // Accumulate transformations to form Q (stored back in V).
    for (int i = 0; i < K; i++) {
        int l = i - 1;
        if (d[i] != 0.0) {
            for (int j = 0; j <= l; j++) {
                double g = 0.0;
                for (int k = 0; k <= l; k++) g += V[i*K + k] * V[k*K + j];
                for (int k = 0; k <= l; k++) V[k*K + j] -= g * V[k*K + i];
            }
        }
        d[i] = V[i*K + i];
        V[i*K + i] = 1.0;
        for (int j = 0; j <= l; j++) V[j*K + i] = V[i*K + j] = 0.0;
    }

    // === Stage 2: Implicit QL with shifts on tridiagonal (NR "tqli") ===
    //
    // d[] holds diagonal, e[] holds subdiagonal (e[0] unused). Drives both
    // to convergence. V is rotated in lockstep so it ends up holding the
    // eigenvectors of the original A.
    for (int i = 1; i < K; i++) e[i - 1] = e[i];
    e[K - 1] = 0.0;

    for (int l = 0; l < K; l++) {
        int iter = 0;
        int m;
        do {
            // Find a small subdiagonal element to split off.
            for (m = l; m < K - 1; m++) {
                double dd = std::fabs(d[m]) + std::fabs(d[m + 1]);
                if (std::fabs(e[m]) + dd == dd) break;
            }
            if (m != l) {
                if (++iter == 100) {
                    // Should never happen for the well-behaved matrices we feed it,
                    // but guard against runaway.
                    break;
                }
                double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
                double r = std::sqrt(g * g + 1.0);
                g = d[m] - d[l] + e[l] / (g + (g >= 0.0 ? r : -r));
                double s_qr = 1.0, c_qr = 1.0, p = 0.0;
                for (int i = m - 1; i >= l; i--) {
                    double f = s_qr * e[i];
                    double b = c_qr * e[i];
                    // Numerically-stable rotation: pick the larger magnitude
                    // to drive division. Matches raxml/coraxlib eigen.c.
                    if (std::fabs(f) >= std::fabs(g)) {
                        c_qr = g / f;
                        r    = std::sqrt(c_qr * c_qr + 1.0);
                        e[i + 1] = f * r;
                        s_qr = 1.0 / r;
                        c_qr *= s_qr;
                    } else {
                        s_qr = f / g;
                        r    = std::sqrt(s_qr * s_qr + 1.0);
                        e[i + 1] = g * r;
                        c_qr = 1.0 / r;
                        s_qr *= c_qr;
                    }
                    g = d[i + 1] - p;
                    r = (d[i] - g) * s_qr + 2.0 * c_qr * b;
                    p = s_qr * r;
                    d[i + 1] = g + p;
                    g = c_qr * r - b;
                    // Rotate eigenvectors (column convention: column i is i-th eigenvector)
                    for (int k = 0; k < K; k++) {
                        double f2 = V[k*K + i + 1];
                        V[k*K + i + 1] = s_qr * V[k*K + i] + c_qr * f2;
                        V[k*K + i]     = c_qr * V[k*K + i] - s_qr * f2;
                    }
                }
                if (r == 0.0 && (m - 1) >= l) continue;
                d[l] -= p;
                e[l]  = g;
                e[m]  = 0.0;
            }
        } while (m != l);
    }
}

GTR::GTR(const double rates[6], const double pi_in[4]) {
    // Normalize pi
    double pi_sum = 0.0;
    for (int i = 0; i < 4; i++) pi_sum += pi_in[i];
    for (int i = 0; i < 4; i++) pi_[i] = pi_in[i] / pi_sum;

    // Build Q from exchangeability rates and pi
    const int pairs[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    double Q[16] = {};
    for (int k = 0; k < 6; k++) {
        int i = pairs[k][0], j = pairs[k][1];
        Q[i * 4 + j] = rates[k] * pi_[j];
        Q[j * 4 + i] = rates[k] * pi_[i];
    }
    // Diagonal: row sum = 0
    for (int i = 0; i < 4; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < 4; j++) if (j != i) row_sum += Q[i * 4 + j];
        Q[i * 4 + i] = -row_sum;
    }
    // Normalize: mean rate = 1
    double scale = 0.0;
    for (int i = 0; i < 4; i++) scale += pi_[i] * (-Q[i * 4 + i]);
    if (scale > 0.0)
        for (int i = 0; i < 16; i++) Q[i] /= scale;
    std::memcpy(Q_, Q, 16 * sizeof(double));

    // Symmetrize: S = diag(sqrt_pi) @ Q @ diag(inv_sqrt_pi)
    double sqrt_pi[4], inv_sqrt_pi[4];
    for (int i = 0; i < 4; i++) {
        sqrt_pi[i]     = std::sqrt(pi_[i]);
        inv_sqrt_pi[i] = 1.0 / sqrt_pi[i];
    }
    double S[16];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            S[i * 4 + j] = sqrt_pi[i] * Q[i * 4 + j] * inv_sqrt_pi[j];

    // Symmetric eigendecomposition of S
    double d[4], U[16];
    symmetric_eigen(S, d, U, 4);
    for (int i = 0; i < 4; i++) eigenvalues_[i] = d[i];

    // L_mat = diag(inv_sqrt_pi) @ U:  L_mat[i][j] = inv_sqrt_pi[i] * U[i][j]
    // R_mat = U^T @ diag(sqrt_pi):    R_mat[i][j] = U[j][i] * sqrt_pi[j]  (transpose)
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            L_mat_[i * 4 + j] = inv_sqrt_pi[i] * U[i * 4 + j];
            R_mat_[i * 4 + j] = U[j * 4 + i] * sqrt_pi[j];
        }
}

void GTR::p_matrix(double t, double* out) const {
    if (t <= 0.0) {
        for (int i = 0; i < 16; i++) out[i] = 0.0;
        for (int i = 0; i < 4; i++) out[i * 4 + i] = 1.0;
        return;
    }

    auto it = cache_.find(t);
    if (it != cache_.end()) {
        COUNTER_INC("p_matrix.gtr.hit");
        std::memcpy(out, it->second.data(), 16 * sizeof(double));
        return;
    }
    COUNTER_INC("p_matrix.gtr.miss");

    // exp_d[j] = exp(eigenvalues[j] * t)
    double exp_d[4];
    for (int j = 0; j < 4; j++) exp_d[j] = std::exp(eigenvalues_[j] * t);

    // P(t) = (L_mat * exp_d[None,:]) @ R_mat
    // Intermediate: tmp[i][j] = L_mat[i][j] * exp_d[j]
    // out[i][k] = sum_j tmp[i][j] * R_mat[j][k]
    double tmp[16];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            tmp[i * 4 + j] = L_mat_[i * 4 + j] * exp_d[j];

    for (int i = 0; i < 4; i++)
        for (int k = 0; k < 4; k++) {
            double s = 0.0;
            for (int j = 0; j < 4; j++) s += tmp[i * 4 + j] * R_mat_[j * 4 + k];
            out[i * 4 + k] = s;
        }

    cache_[t] = std::vector<double>(out, out + 16);
}

// ================================================================== //
// GTR::dp_ddp_matrix                                                  //
// ================================================================== //

void GTR::dp_ddp_matrix(double t, double* dp, double* ddp) const {
    if (t <= 0.0) t = 1e-10;
    // exp_d[j] = exp(lambda[j] * t)
    double exp_d[4];
    for (int j = 0; j < 4; j++) exp_d[j] = std::exp(eigenvalues_[j] * t);

    // dP/dt  = (L_mat * (lambda   * exp_d)) @ R_mat
    // d2P/dt2 = (L_mat * (lambda^2 * exp_d)) @ R_mat
    double tmp1[16], tmp2[16];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            tmp1[i*4+j] = L_mat_[i*4+j] * eigenvalues_[j]               * exp_d[j];
            tmp2[i*4+j] = L_mat_[i*4+j] * eigenvalues_[j]*eigenvalues_[j] * exp_d[j];
        }
    for (int i = 0; i < 4; i++)
        for (int k = 0; k < 4; k++) {
            double s1 = 0.0, s2 = 0.0;
            for (int j = 0; j < 4; j++) {
                s1 += tmp1[i*4+j] * R_mat_[j*4+k];
                s2 += tmp2[i*4+j] * R_mat_[j*4+k];
            }
            dp[i*4+k]  = s1;
            ddp[i*4+k] = s2;
        }
}
