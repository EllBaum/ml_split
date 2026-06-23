// jck.h - K-state Jukes-Cantor model (JC69 = K4, Poisson = K20).

#pragma once

#include "subst_model.h"
#include <vector>
#include <unordered_map>

// ------------------------------------------------------------------ //
// JCK — Jukes-Cantor generalised to K states.                        //
//   K=4  → JC69 (DNA)                                                //
//   K=20 → Poisson (AA)                                              //
// Analytic P(t), no matrix exponentiation.                            //
// P[i,i](t) = (1 + (K-1)*exp(-K*t/(K-1))) / K                       //
// P[i,j](t) = (1 -        exp(-K*t/(K-1))) / K   i != j             //
// ------------------------------------------------------------------ //

class JCK : public SubstModel {
public:
    explicit JCK(int k = 4);
    int    K()    const override { return K_; }
    const double* pi() const override { return pi_.data(); }
    void   p_matrix(double t, double* out) const override;
    void   dp_ddp_matrix(double t, double* dp, double* ddp) const;

    // JC has an analytic eigendecomposition (uniform pi, equal exchange rates).
    // Exposed so the sumtable BLO path works for JC/JCAA exactly as it does for
    // GTR/JTT.  P(t) = L_mat · diag(exp(eigenvalues·t)) · R_mat.
    const double* eigenvalues() const override { return eigenvalues_; }
    const double* L_mat()       const override { return L_mat_; }
    const double* R_mat()       const override { return R_mat_; }

private:
    // Build eigenvalues_/L_mat_/R_mat_ once in the constructor, using the same
    // symmetrize → eigensolve → L/R construction as JTT/GTR (convention parity).
    void _setup_eigen();

    int K_;
    std::vector<double> pi_;
    // Fixed in-object eig storage (max K = 20, JCAA), matching EmpiricalAAModel
    // and GTR. NB: using std::vector here would add heap allocations that
    // perturb the process-wide allocation layout; the empirical models use
    // fixed arrays for exactly this reason, so JCK matches them.
    double eigenvalues_[20];   // K
    double L_mat_[400];        // K*K, row-major
    double R_mat_[400];        // K*K, row-major
    mutable std::unordered_map<double, std::vector<double>> cache_;
};

// Convenience aliases
using JC69   = JCK;   // K=4  (default)
using Poisson = JCK;  // K=20


// JCModel: alias for JCK
using JCModel = JCK;
