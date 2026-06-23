// empirical_aa_model.h - empirical amino-acid model base class.

#pragma once

#include "subst_model.h"
#include <vector>
#include <unordered_map>

// ------------------------------------------------------------------ //
// EmpiricalAAModel — empirical amino acid substitution model.         //
//   Accepts a 190-value upper-triangle exchangeability array and      //
//   20 stationary frequencies. Builds Q, normalises, eigendecomposes //
//   once at construction. P(t) and derivatives are O(K^2) per call.  //
//   K=20 fixed. AA order: ACDEFGHIKLMNPQRSTVWY (alphabetical).       //
// ------------------------------------------------------------------ //

class EmpiricalAAModel : public SubstModel {
public:
    // exch[190]: upper-triangle exchangeabilities (i<j, row-major).
    // freqs[20]: stationary frequencies (will be normalised).
    EmpiricalAAModel(const double exch[190], const double freqs[20]);

    int    K()    const override { return 20; }
    const double* pi() const override { return pi_; }
    void   p_matrix    (double t, double* out)              const override;
    void   dp_ddp_matrix(double t, double* dp, double* ddp) const;

    const double* eigenvalues() const override { return eigenvalues_; }
    const double* L_mat()       const override { return L_mat_; }
    const double* R_mat()       const override { return R_mat_; }

private:
    static constexpr int K_ = 20;
    double pi_[20];
    double eigenvalues_[20];
    double L_mat_[400];   // diag(inv_sqrt_pi) @ U,  K*K row-major
    double R_mat_[400];   // U^T @ diag(sqrt_pi),    K*K row-major
    mutable std::unordered_map<double, std::vector<double>> cache_;
};
