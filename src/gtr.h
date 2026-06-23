// gtr.h - General Time-Reversible model + shared symmetric eigensolver.

#pragma once

#include "subst_model.h"
#include <vector>
#include <unordered_map>

// ------------------------------------------------------------------ //
// GTR — General Time-Reversible. Eigendecomposition, O(K^2) per t.  //
// ------------------------------------------------------------------ //

class GTR : public SubstModel {
public:
    // rates[6] = [AC, AG, AT, CG, CT, GT] exchangeability parameters.
    // pi[4]    = stationary frequencies (will be normalized).
    GTR(const double rates[6], const double pi[4]);

    int    K()    const override { return 4; }
    const double* pi() const override { return pi_; }
    // P(t) = L_mat * exp(lambda*t) @ R_mat
    void   p_matrix(double t, double* out) const override;
    // dP/dt  = (L_mat * (lambda   * exp_d)) @ R_mat
    // d2P/dt2 = (L_mat * (lambda^2 * exp_d)) @ R_mat
    void   dp_ddp_matrix(double t, double* dp, double* ddp) const;

    const double* eigenvalues() const override { return eigenvalues_; }
    const double* L_mat()       const override { return L_mat_; }
    const double* R_mat()       const override { return R_mat_; }

private:
    double pi_[4];
    double Q_[16];           // rate matrix (4*4, row-major), for reference
    double eigenvalues_[4];  // eigenvalues of symmetrized Q
    double L_mat_[16];       // diag(inv_sqrt_pi) @ U,  K*K row-major
    double R_mat_[16];       // U^T @ diag(sqrt_pi),    K*K row-major
    mutable std::unordered_map<double, std::vector<double>> cache_;

public:
    // Eigendecomposition for real symmetric K*K matrix.
    // Public so EmpiricalAAModel can reuse it.
    // On entry: A[K*K] = symmetric matrix. On exit: A diagonalized,
    // d[K] = eigenvalues, V[K*K] = eigenvectors in columns (row-major).
    static void symmetric_eigen(double* A, double* d, double* V, int K);
};
