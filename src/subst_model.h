// subst_model.h - abstract substitution-model interface.

#pragma once

// ------------------------------------------------------------------ //
// SubstModel — abstract substitution model interface                  //
// ------------------------------------------------------------------ //

class SubstModel {
public:
    virtual ~SubstModel() = default;
    virtual int    K()    const = 0;
    virtual const double* pi() const = 0;
    // Fill out[K*K] with P(t), row-major. out is pre-allocated by caller.
    virtual void   p_matrix(double t, double* out) const = 0;
    // Fill dp[K*K] = dP/dt and ddp[K*K] = d²P/dt². Both pre-allocated.
    virtual void   dp_ddp_matrix(double t, double* dp, double* ddp) const = 0;

    // Eigendecomposition accessors for sumtable BLO. Convention:
    //   P(t) = L_mat · diag(exp(eigenvalues · t)) · R_mat
    // Models without an eigendecomposition return nullptr; sumtable BLO then
    // falls back to the old _p_apply×3 kernel.
    virtual const double* eigenvalues() const { return nullptr; }
    virtual const double* L_mat()       const { return nullptr; }
    virtual const double* R_mat()       const { return nullptr; }
};
