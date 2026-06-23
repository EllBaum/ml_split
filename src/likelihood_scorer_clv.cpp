// likelihood_scorer_clv.cpp — part of the likelihood_scorer_unit.cpp ubmrella.

#include "likelihood_scorer.h"
#line 1298 "likelihood_scorer.cpp"

void LikelihoodScorer::_clv(int node, int excl,
                             double* out_clv, double* out_log_scale) const {
    PROF_SCOPE("_clv");
    ++clv_calc_count_;
    int KL = K_ * L_;

    if (toward_[node] == excl) {
        if (g_clv_depth == 0) COUNTER_INC("clv.toward_memcpy_top");
        else                  COUNTER_INC("clv.toward_memcpy_nested");
        COUNTER_INC("clv.toward_memcpy");
        int s = _slot_of(node, excl);
        std::memcpy(out_clv, _clv_slot(node, s),
                    KL * sizeof(double));
        std::memcpy(out_log_scale, &log_scale_arr_[(node * 3 + s) * L_],
                    L_ * sizeof(double));
        return;
    }
    if (tree_->is_leaf(node)) {
        COUNTER_INC("clv.leaf_memcpy");
        std::memcpy(out_clv, &tip_clv_[node * KL], KL * sizeof(double));
        std::fill(out_log_scale, out_log_scale + L_, 0.0);
        return;
    }

    int children[2];
    int nc = tree_->neighbors_except(node, excl, children);
    if (nc != 2) {
        // excl not a neighbor — fallback to precomputed (toward slot)
        COUNTER_INC("clv.fallback_toward");
        int s = _slot_of(node, toward_[node]);
        std::memcpy(out_clv, _clv_slot(node, s),
                    KL * sizeof(double));
        std::memcpy(out_log_scale, &log_scale_arr_[(node * 3 + s) * L_],
                    L_ * sizeof(double));
        return;
    }

    // Non-toward direction: read the clean cached slot directly (filled by the
    // down-pass), skipping recursion; dirty slots fall through to the recursive
    // recompute below.
    {
        int s = _slot_of(node, excl);
        if (s >= 0 && !dirty_[node * 3 + s]) {
            COUNTER_INC("clv.stage3_lookup");
            std::memcpy(out_clv, _clv_slot(node, s),
                        KL * sizeof(double));
            std::memcpy(out_log_scale,
                        &log_scale_arr_[(node * 3 + s) * L_],
                        L_ * sizeof(double));
            return;
        }
        COUNTER_INC("clv.stage3_dirty_fallback");
    }

    // Recursive descent: depth-indexed thread_local scratch for recursive
    // children; toward/leaf children use slot pointers directly (zero-copy).
    COUNTER_INC("clv.recursive");
    int c1 = children[0], c2 = children[1];
    int KK = K_ * K_;
    g_clv_scratch.ensure(g_clv_depth, KL, L_, KK);
    double* r1_buf = g_clv_scratch.r1 [g_clv_depth].data();
    double* r2_buf = g_clv_scratch.r2 [g_clv_depth].data();
    double* ls1_buf = g_clv_scratch.ls1[g_clv_depth].data();
    double* ls2_buf = g_clv_scratch.ls2[g_clv_depth].data();
    double* P1     = g_clv_scratch.P1 [g_clv_depth].data();
    double* P2     = g_clv_scratch.P2 [g_clv_depth].data();

    // Child CLV by pointer when possible (toward/leaf); else _clv into scratch.
    const double* r1; const double* ls1;
    const double* r2; const double* ls2;
    auto get_child = [&](int child, double* clv_scratch, double* ls_scratch,
                         const double*& clv_out, const double*& ls_out) {
        if (tree_->is_leaf(child)) {
            COUNTER_INC("clv.leaf_pointer");
            clv_out = &tip_clv_[child * KL];
            std::fill(ls_scratch, ls_scratch + L_, 0.0);
            ls_out = ls_scratch;
            return;
        }
        if (toward_[child] == node) {
            COUNTER_INC("clv.toward_pointer");
            int s = _slot_of(child, node);
            clv_out = _clv_slot(child, s);
            ls_out  = &log_scale_arr_[(child * 3 + s) * L_];
            return;
        }
        // Recursive: child is non-toward of node — must compute.
        g_clv_depth++;
        _clv(child, node, clv_scratch, ls_scratch);
        g_clv_depth--;
        clv_out = clv_scratch;
        ls_out  = ls_scratch;
    };
    get_child(c1, r1_buf, ls1_buf, r1, ls1);
    get_child(c2, r2_buf, ls2_buf, r2, ls2);
    _p_for_edge(node, c1, P1);
    _p_for_edge(node, c2, P2);

    // Tip kernels: leaf CLVs are sparse 0/1, so use the lookup-table kernel
    // instead of a full dgemv.
    const bool tip_kernel = _tip_kernel_enabled();
    const bool c1_tip = tip_kernel && tree_->is_leaf(c1);
    const bool c2_tip = tip_kernel && tree_->is_leaf(c2);

    if (!c1_tip && !c2_tip) {
        // Internal-internal: fused pair kernel.
        COUNTER_INC("clv.kernel.pair_pair");
        _p_apply_pair(P1, r1, P2, r2, out_clv);
    } else if (c1_tip && c2_tip) {
        // Both tips: fused tip-tip kernel.
        COUNTER_INC("clv.kernel.tip_tip");
        _p_apply_tip_tip_pair(P1, bl_[node * n_ + c1], c1,
                              P2, bl_[node * n_ + c2], c2,
                              out_clv);
    } else {
        // One tip, one internal: fused kernel (symmetric in which side is tip).
        COUNTER_INC("clv.kernel.tip_internal");
        if (c1_tip) _p_apply_tip_internal_pair(
                        P1, bl_[node * n_ + c1], c1,
                        P2, r2, out_clv);
        else        _p_apply_tip_internal_pair(
                        P2, bl_[node * n_ + c2], c2,
                        P1, r1, out_clv);
    }

    for (int l = 0; l < L_; l++) out_log_scale[l] = ls1[l] + ls2[l];
    _scale_clv_inplace(out_clv, out_log_scale);
}

// Pointer view of _clv: zero-copy for toward/leaf; recursive case writes into
// the caller's scratch.
void LikelihoodScorer::_clv_view(int node, int excl,
                                  const double** out_clv_ptr,
                                  const double** out_ls_ptr,
                                  double* scratch_clv,
                                  double* scratch_ls) const {
    int KL = K_ * L_;
    if (toward_[node] == excl) {
        COUNTER_INC("clv_view.toward_pointer_top");
        int s = _slot_of(node, excl);
        *out_clv_ptr = _clv_slot(node, s);
        *out_ls_ptr  = &log_scale_arr_[(node * 3 + s) * L_];
        return;
    }
    if (tree_->is_leaf(node)) {
        COUNTER_INC("clv_view.leaf_pointer_top");
        *out_clv_ptr = &tip_clv_[node * KL];
        std::fill(scratch_ls, scratch_ls + L_, 0.0);
        *out_ls_ptr  = scratch_ls;
        return;
    }
    // Non-toward: must compute. Write into scratch.
    COUNTER_INC("clv_view.recursive_top");
    _clv(node, excl, scratch_clv, scratch_ls);
    *out_clv_ptr = scratch_clv;
    *out_ls_ptr  = scratch_ls;
}

double LikelihoodScorer::_pseudoroot_loglik() const {
    int KL = K_ * L_;
    int cs = calc_start_;
    std::vector<double> P0(K_ * K_), leaf0_contrib(KL), root_clv(KL);
    std::vector<double> root_ls(L_);

    _p_for_edge(cs, 0, P0.data());
    _p_apply(P0.data(), &tip_clv_[0], leaf0_contrib.data());
    // CLV at cs looking AWAY from leaf 0: that's the slot at cs whose excl is leaf 0.
    int s_cs = _slot_of(cs, 0);
    _clv_mul(leaf0_contrib.data(),
             _clv_slot(cs, s_cs), root_clv.data());
    std::memcpy(root_ls.data(),
                &log_scale_arr_[(cs * 3 + s_cs) * L_],
                L_ * sizeof(double));
    _scale_clv_inplace(root_clv.data(), root_ls.data());

    return _total_loglik(root_clv.data(), root_ls.data());
}

double LikelihoodScorer::_total_loglik(const double* root_clv,
                                        const double* log_scale) const {
    const double* pi = model_->pi();
    const int*    w  = msa_->weights.data();
    int K = K_, L = L_;
    double ll = 0.0;
    for (int l = 0; l < L; l++) {
        double s = 0.0;
        for (int k = 0; k < K; k++) s += pi[k] * root_clv[k * L + l];
        if (s < 1e-300) s = 1e-300;
        ll += static_cast<double>(w[l]) * (std::log(s) + log_scale[l]);
    }
    return ll;
}

// ================================================================== //
// _build_tip_clvs                                                     //
// ================================================================== //

void LikelihoodScorer::_build_tip_clvs() {
    int K = K_, L = L_;
    int n_taxa = tree_->n_taxa;
    // Bit i in the MSA mask -> tip_clv[k=i]=1.0; gap/all-zero -> all 1.0.
    for (int t = 0; t < n_taxa; t++) {
        double* dst = &tip_clv_[t * K * L];
        for (int l = 0; l < L; l++) {
            uint32_t mask = msa_->at(t, l);
            bool any = false;
            for (int k = 0; k < K; k++) {
                double v = ((mask >> k) & 1u) ? 1.0 : 0.0;
                dst[k * L + l] = v;
                if (v > 0.0) any = true;
            }
            if (!any)
                for (int k = 0; k < K; k++) dst[k * L + l] = 1.0;
        }
    }

    // Compact tips: per (taxon, pattern) a uint8_t code into a table of unique
    // bitmasks, so the tip kernels do a lookup instead of a K*K dgemv. Gap
    // (mask 0) remaps to all-set, matching the dense-CLV all-1.0 gap.
    tip_codes_.assign(n_taxa * L, 0);
    tip_mask_table_.clear();
    std::unordered_map<uint32_t, uint8_t> mask_to_code;
    const uint32_t all_mask = (K < 32) ? ((1u << K) - 1u) : 0xFFFFFFFFu;

    for (int t = 0; t < n_taxa; t++) {
        for (int l = 0; l < L; l++) {
            uint32_t mask = msa_->at(t, l) & all_mask;  // strip bits >= K
            if (mask == 0) mask = all_mask;             // gap -> all-set
            auto it = mask_to_code.find(mask);
            uint8_t code;
            if (it == mask_to_code.end()) {
                // 255 unique codes is plenty for AA (~25) and DNA (16).
                assert(tip_mask_table_.size() < 255 &&
                       "tip mask table overflowed uint8_t code range");
                code = (uint8_t)tip_mask_table_.size();
                tip_mask_table_.push_back(mask);
                mask_to_code[mask] = code;
            } else {
                code = it->second;
            }
            tip_codes_[t * L + l] = code;
        }
    }
    n_tip_codes_ = (int)tip_mask_table_.size();
}

// ================================================================== //
// _refresh_toward                                                     //
// ================================================================== //

void LikelihoodScorer::_refresh_toward() {
    const Tree& tree = *tree_;
    struct Frame { int node, came_from; };
    std::stack<Frame> stk;
    stk.push({calc_start_, 0});
    while (!stk.empty()) {
        auto [node, cf] = stk.top(); stk.pop();
        toward_[node] = cf;
        int nbs[3];
        int nc = tree.neighbors_of(node, nbs);
        for (int i = 0; i < nc; i++)
            if (nbs[i] != cf)
                stk.push({nbs[i], node});
    }
    toward_[0] = calc_start_;
}


// ================================================================== //
// _full_recompute                                                     //
// ================================================================== //

void LikelihoodScorer::_full_recompute() {
    PROF_SCOPE("_full_recompute");
    const Tree& tree = *tree_;
    int KL = K_ * L_, KK = K_ * K_;

    _refresh_toward();

    // Mark all slots dirty first; a prior toward_ assignment can otherwise
    // leave stale slots reading clean.
    std::fill(dirty_.begin(), dirty_.end(), 1);

    // Rebuild every p_arr slot (mostly model hash-cache hits + memcpy).
    for (int v = 0; v < n_; v++) {
        int base = v * 3;
        for (int s = 0; s < 3; s++) {
            int nb = tree.neighbors[base + s];
            if (nb == INVALID) continue;
            double* p_slot = &p_arr_[(v * 3 + s) * KK];
            model_->p_matrix(bl_[v * n_ + nb], p_slot);
        }
    }

    // Seed leaf CLVs (incl. node 0); each leaf uses only slot 0.
    for (int v = 0; v < tree.n_taxa; v++) {
        std::fill(&log_scale_arr_[(v * 3 + 0) * L_],
                  &log_scale_arr_[(v * 3 + 0) * L_] + L_, 0.0);
    }

    // Up-pass: fill each v's toward slot (CLV looking away from toward[v]).
    std::vector<int> order = tree.postorder(calc_start_, 0);
    std::vector<double> a1(KL), a2(KL);

    for (int v : order) {
        if (tree.is_leaf(v)) continue;

        int children[2];
        tree.neighbors_except(v, toward_[v], children);
        int c1 = children[0], c2 = children[1];

        // Each child's CLV-away-from-v sits in its toward slot (toward[c]==v
        // here), and the (c,v) p_arr in p_arr_[c, slot_of(c, v)].
        int s_c1 = tree.is_leaf(c1) ? 0 : _slot_of(c1, v);
        int s_c2 = tree.is_leaf(c2) ? 0 : _slot_of(c2, v);

        _p_apply(&p_arr_[(c1 * 3 + s_c1) * KK],
                 _clv_slot(c1, s_c1), a1.data());

        int s_v = _slot_of(v, toward_[v]);
        double* dst = _clv_slot(v, s_v);
        double* ls  = &log_scale_arr_[(v * 3 + s_v) * L_];
        _p_apply_mul(&p_arr_[(c2 * 3 + s_c2) * KK],
                     _clv_slot(c2, s_c2), a1.data(), dst);

        const double* ls1 = &log_scale_arr_[(c1 * 3 + s_c1) * L_];
        const double* ls2 = &log_scale_arr_[(c2 * 3 + s_c2) * L_];
        for (int l = 0; l < L_; l++) ls[l] = ls1[l] + ls2[l];
        _scale_clv_inplace(dst, ls);

        dirty_[v * 3 + s_v] = 0;
    }

    for (int v = 0; v < tree.n_taxa; v++) dirty_[v * 3 + 0] = 0;

    // Down-pass (preorder from calc_start): fill the two non-toward
    // slots of each internal node. The slot at v away from child ci is the
    // product of the toward contribution P(bl[v,t]) @ slot[t->v] and the other
    // child's P(bl[v,c_other]) @ slot[c_other->v]. Preorder guarantees
    // slot[t->v] is already populated when v is processed.
    {
        std::vector<double> contrib_t(KL), contrib_o(KL);
        std::stack<std::pair<int, int>> stk;
        stk.push({calc_start_, -1});
        while (!stk.empty()) {
            auto [v, /*cf*/_unused] = stk.top(); stk.pop();
            if (tree.is_leaf(v)) continue;

            int t = toward_[v];
            int kids[2];
            int nc = tree.neighbors_except(v, t, kids);
            if (nc != 2) continue;

            // contribution from toward direction (shared by both non-toward slots)
            int s_t_to_v = _slot_of(t, v);
            int s_v_to_t = _slot_of(v, t);
            const double* clv_t = _clv_slot(t, s_t_to_v);
            const double* ls_t  = &log_scale_arr_[(t * 3 + s_t_to_v) * L_];
            const double* P_v_t = &p_arr_[(v * 3 + s_v_to_t) * KK];
            _p_apply(P_v_t, clv_t, contrib_t.data());

            for (int i = 0; i < 2; i++) {
                int c       = kids[i];
                int c_other = kids[1 - i];

                int s_co_to_v = _slot_of(c_other, v);
                int s_v_to_co = _slot_of(v, c_other);
                const double* clv_co = _clv_slot(c_other, s_co_to_v);
                const double* ls_co  = &log_scale_arr_[(c_other * 3 + s_co_to_v) * L_];
                const double* P_v_co = &p_arr_[(v * 3 + s_v_to_co) * KK];

                int s_v_to_c = _slot_of(v, c);
                double* dst    = _clv_slot(v, s_v_to_c);
                double* dst_ls = &log_scale_arr_[(v * 3 + s_v_to_c) * L_];
                _p_apply_mul(P_v_co, clv_co, contrib_t.data(), dst);
                for (int l = 0; l < L_; l++) dst_ls[l] = ls_t[l] + ls_co[l];
                _scale_clv_inplace(dst, dst_ls);

                dirty_[v * 3 + s_v_to_c] = 0;
            }

            for (int i = 0; i < 2; i++) {
                if (!tree.is_leaf(kids[i])) stk.push({kids[i], v});
            }
        }
    }

    score_ = _pseudoroot_loglik();

    // CLV_VERIFY: compare every populated slot against _clv_truth.
    if (const char* env = std::getenv("CLV_VERIFY")) {
        if (env[0] && env[0] != '0') _verify_clv_truth();
    }
}