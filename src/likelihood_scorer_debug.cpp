// likelihood_scorer_debug.cpp — CLV ground-truth oracle + verifier (CLV_VERIFY,
// diagnostic only). Part of the likelihood_scorer_unit.cpp umbrella.
#include "likelihood_scorer.h"

// ================================================================== //
// _clv_truth                                                          //
// ================================================================== //
// Diagnostic ground truth: recurse CLV from tip_clv_ + bl_ ONLY, all the way
// down — never reads clv_arr_/p_arr_, so it can't inherit a stale slot. P is
// computed fresh from bl_ via model_->p_matrix() (value-keyed cache). Same
// scaling rule as production (_scale_clv_inplace per internal frame); leaves
// give tip_clv_ + log_scale 0. Per-frame heap allocs (perf irrelevant).
void LikelihoodScorer::_clv_truth(int node, int excl,
                                  double* out_clv, double* out_ls) const {
    const int KL = K_ * L_;
    const int KK = K_ * K_;

    if (tree_->is_leaf(node)) {
        std::memcpy(out_clv, &tip_clv_[node * KL], KL * sizeof(double));
        std::fill(out_ls, out_ls + L_, 0.0);
        return;
    }

    int children[2];
    int nc = tree_->neighbors_except(node, excl, children);
    if (nc != 2) {
        // excl not a neighbor: undefined CLV — fill NaN so a comparison flags it.
        std::fill(out_clv, out_clv + KL,
                  std::numeric_limits<double>::quiet_NaN());
        std::fill(out_ls, out_ls + L_,
                  std::numeric_limits<double>::quiet_NaN());
        return;
    }
    int c1 = children[0], c2 = children[1];

    std::vector<double> r1(KL), r2(KL), ls1(L_), ls2(L_);
    std::vector<double> P1(KK), P2(KK), a1(KL), a2(KL);

    _clv_truth(c1, node, r1.data(), ls1.data());
    _clv_truth(c2, node, r2.data(), ls2.data());

    // Fresh P matrices from current bl_ — bypass p_arr_ entirely.
    model_->p_matrix(bl_[node * n_ + c1], P1.data());
    model_->p_matrix(bl_[node * n_ + c2], P2.data());

    _p_apply(P1.data(), r1.data(), a1.data());
    _p_apply(P2.data(), r2.data(), a2.data());
    _clv_mul(a1.data(), a2.data(), out_clv);
    for (int l = 0; l < L_; l++) out_ls[l] = ls1[l] + ls2[l];
    _scale_clv_inplace(out_clv, out_ls);
}

// ================================================================== //
// _verify_clv_truth                                                   //
// ================================================================== //
// Walk every internal node v. For each populated (non-dirty) slot, compare it
// against _clv_truth; also compare what recursive _clv() returns (what BLO
// actually reads) for every direction against truth. A scale-step mismatch
// shows up as |log_scale_diff| ≈ |LOG_SCALE| (~92.1), not as a tiny clv_diff.
void LikelihoodScorer::_verify_clv_truth() const {
    const int KL = K_ * L_;
    int n_mismatch_slot = 0, n_mismatch_recursive = 0;
    double max_slot_clv_diff = 0.0, max_slot_ls_diff = 0.0;
    double max_rec_clv_diff = 0.0,  max_rec_ls_diff = 0.0;
    int worst_slot_node = -1, worst_rec_node = -1, worst_rec_excl = -1;

    std::vector<double> truth_clv(KL), truth_ls(L_);
    std::vector<double> rec_clv(KL),   rec_ls(L_);

    for (int v = tree_->n_taxa; v < n_; v++) {
        // Only non-dirty slots — i.e. what was populated this call. dirty_ is
        // reset all-1 at entry, so stale slots from prior topologies are skipped.
        for (int s = 0; s < 3; s++) {
            int nbr = tree_->neighbors[v * 3 + s];
            if (nbr == INVALID) continue;
            if (dirty_[v * 3 + s]) continue;

            _clv_truth(v, nbr, truth_clv.data(), truth_ls.data());
            const double* slot_clv = _clv_slot(v, s);
            const double* slot_ls  = &log_scale_arr_[(v * 3 + s) * L_];

            double mclv = 0.0, mls = 0.0;
            for (int i = 0; i < KL; i++) {
                double d = std::fabs(slot_clv[i] - truth_clv[i]);
                if (d > mclv) mclv = d;
            }
            for (int l = 0; l < L_; l++) {
                double d = std::fabs(slot_ls[l] - truth_ls[l]);
                if (d > mls) mls = d;
            }
            if (mclv > 1e-9 || mls > 1e-9) {
                n_mismatch_slot++;
                if (mclv > max_slot_clv_diff) {
                    max_slot_clv_diff = mclv;
                    worst_slot_node   = v;
                }
                if (mls > max_slot_ls_diff) max_slot_ls_diff = mls;
            }
        }

        // Compare what recursive _clv() returns (what BLO actually reads)
        // against truth, for all 3 directions at v.
        for (int s = 0; s < 3; s++) {
            int nbr = tree_->neighbors[v * 3 + s];
            if (nbr == INVALID) continue;
            _clv(v, nbr, rec_clv.data(), rec_ls.data());
            _clv_truth(v, nbr, truth_clv.data(), truth_ls.data());
            double mclv = 0.0, mls = 0.0;
            for (int i = 0; i < KL; i++) {
                double d = std::fabs(rec_clv[i] - truth_clv[i]);
                if (d > mclv) mclv = d;
            }
            for (int l = 0; l < L_; l++) {
                double d = std::fabs(rec_ls[l] - truth_ls[l]);
                if (d > mls) mls = d;
            }
            if (mclv > 1e-9 || mls > 1e-9) {
                n_mismatch_recursive++;
                if (mclv > max_rec_clv_diff) {
                    max_rec_clv_diff = mclv;
                    worst_rec_node   = v;
                    worst_rec_excl   = nbr;
                }
                if (mls > max_rec_ls_diff) max_rec_ls_diff = mls;
            }
        }
    }

    std::cerr << "[CLV_VERIFY] populated-slot mismatches: " << n_mismatch_slot
              << " (max clv_diff=" << max_slot_clv_diff
              << " max ls_diff="   << max_slot_ls_diff
              << " worst node=" << worst_slot_node << ")\n";
    std::cerr << "[CLV_VERIFY] recursive-_clv  mismatches: " << n_mismatch_recursive
              << " (max clv_diff=" << max_rec_clv_diff
              << " max ls_diff="   << max_rec_ls_diff
              << " worst node="    << worst_rec_node
              << " excl="          << worst_rec_excl << ")\n";
}
