// likelihood_scorer_commit.cpp — part of the likelihood_scorer_unit.cpp umbrella.
#include "likelihood_scorer.h"
#line 3073 "likelihood_scorer.cpp"
void LikelihoodScorer::_propagate_up_ll(int start,
                                         std::unordered_set<int>& visited) {
    PROF_SCOPE("_propagate_up_ll");
    const Tree& tree = *tree_;
    int KL = K_ * L_, KK = K_ * K_;
    std::vector<double> new_raw(KL), a1(KL), a2(KL);
    std::vector<double> new_ls(L_);

    int node = start;
    while (true) {
        if (tree.is_leaf(node) || node == 0) break;
        int toward_node = toward_[node];

        int children[2];
        tree.neighbors_except(node, toward_node, children);
        int c1 = children[0], c2 = children[1];

        int s_c1 = tree.is_leaf(c1) ? 0 : _slot_of(c1, node);
        int s_c2 = tree.is_leaf(c2) ? 0 : _slot_of(c2, node);

        // _p_apply_pair (not _clv_mul on contribs) to match _clv's
        // internal-internal kernel sequence bit-for-bit. a2 unused.
        (void)a2;
        _p_apply_pair(&p_arr_[(c1 * 3 + s_c1) * KK],
                      _clv_slot(c1, s_c1),
                      &p_arr_[(c2 * 3 + s_c2) * KK],
                      _clv_slot(c2, s_c2),
                      new_raw.data());
        const double* ls1 = &log_scale_arr_[(c1 * 3 + s_c1) * L_];
        const double* ls2 = &log_scale_arr_[(c2 * 3 + s_c2) * L_];
        for (int l = 0; l < L_; l++) new_ls[l] = ls1[l] + ls2[l];
        _scale_clv_inplace(new_raw.data(), new_ls.data());

        int s_node = _slot_of(node, toward_node);

        // Stop early if this node was already refreshed and is unchanged.
        bool in_visited = visited.count(node) > 0;
        if (in_visited) {
            const double* old_clv = _clv_slot(node, s_node);
            const double* old_ls  = &log_scale_arr_[(node * 3 + s_node) * L_];
            bool same = true;
            for (int i = 0; i < KL && same; i++)
                if (std::fabs(new_raw[i] - old_clv[i]) >
                    1e-10 * std::fabs(old_clv[i]) + 0.0) same = false;
            if (same)
                for (int l = 0; l < L_ && same; l++)
                    if (std::fabs(new_ls[l] - old_ls[l]) > 1e-12) same = false;
            if (same) break;
        }

        visited.insert(node);
        std::memcpy(_clv_slot(node, s_node),
                    new_raw.data(), KL * sizeof(double));
        std::memcpy(&log_scale_arr_[(node * 3 + s_node) * L_],
                    new_ls.data(),  L_ * sizeof(double));
        dirty_[node * 3 + s_node] = 0;
        node = toward_node;
    }
}

// ================================================================== //
// commit — incremental                                                //
// ================================================================== //

void LikelihoodScorer::commit(const LikelihoodSPRCandidate& cand) {
    PROF_SCOPE("commit");

    // LOG_COMMITS (env, default 0=silent): 1=prune/regraft+LL, 2=+depth+triplet
    // BLs, 3=+path nodes.
    static int log_level = []() {
        const char* s = std::getenv("LOG_COMMITS");
        return s ? std::atoi(s) : 0;
    }();
    if (log_level >= 1) {
        std::fprintf(stderr,
            "COMMIT pu=%d pv=%d ra=%d rb=%d ll=%.6f",
            cand.prune_u, cand.prune_v, cand.ra, cand.rb,
            cand.log_likelihood);
        if (log_level >= 2) {
            std::fprintf(stderr, " depth=%zu",
                         cand.path_nodes.size());
            if (!std::isnan(cand.triplet_bl_pendant)) {
                std::fprintf(stderr, " bl_pend=%.6f bl_ra=%.6f bl_rb=%.6f",
                    cand.triplet_bl_pendant,
                    cand.triplet_bl_ra,
                    cand.triplet_bl_rb);
            }
        }
        if (log_level >= 3) {
            std::fprintf(stderr, " path=[");
            for (size_t i = 0; i < cand.path_nodes.size(); i++)
                std::fprintf(stderr, "%s%d", i ? "," : "", cand.path_nodes[i]);
            std::fprintf(stderr, "]");
        }
        std::fprintf(stderr, "\n");
    }

    int pu  = cand.prune_u;
    int ra  = cand.ra,  rb  = cand.rb;
    int nb1 = cand.nb1, nb2 = cand.nb2;
    int KL  = K_ * L_, KK  = K_ * K_;

    // Conservative dirty reset: an SPR changes the frontier subtree, dirtying
    // 2 of 3 slots at every node. Rather than find the one clean slot per node,
    // mark all dirty; _propagate_up_ll re-clears the toward slots it refreshes,
    // the rest fall through to recursion. The next _full_recompute (end of each
    // FAST BLO sweep) repopulates everything.
    std::fill(dirty_.begin(), dirty_.end(), 1);

    double old_nb1_pu = bl_[nb1 * n_ + pu];
    double old_nb2_pu = bl_[nb2 * n_ + pu];
    double old_ra_rb  = bl_[ra  * n_ + rb];

    // BL convention: merged nb1—nb2 edge gets the sum; the two new pu sub-edges
    // each get half of ra—rb. If triplet BLO ran, the candidate carries
    // optimized BLs for pendant + both sub-edges; use those instead.
    bl_[nb1 * n_ + nb2] = bl_[nb2 * n_ + nb1] = old_nb1_pu + old_nb2_pu;

    bool have_triplet_bls = !std::isnan(cand.triplet_bl_ra)
                         && !std::isnan(cand.triplet_bl_rb)
                         && !std::isnan(cand.triplet_bl_pendant);
    double new_pu_ra      = have_triplet_bls ? cand.triplet_bl_ra      : (old_ra_rb * 0.5);
    double new_pu_rb      = have_triplet_bls ? cand.triplet_bl_rb      : (old_ra_rb * 0.5);
    double new_pu_pv      = have_triplet_bls ? cand.triplet_bl_pendant : bl_[pu * n_ + cand.prune_v];

    bl_[ra * n_ + pu] = bl_[pu * n_ + ra] = new_pu_ra;
    bl_[rb * n_ + pu] = bl_[pu * n_ + rb] = new_pu_rb;
    bl_[pu * n_ + cand.prune_v] = bl_[cand.prune_v * n_ + pu] = new_pu_pv;

    // 5-BLO override for the 2 extra edges around the post-SPR NNI center.
    // Post-SPR ra's neighbors are {pu, nb_other_pre, ra_other}; ra_other is the
    // one excluding pu AND nb_other_pre (not rb — rb is now pu's neighbor, so
    // excluding it would let nb_other_pre be picked and corrupt the edge).
    if (!std::isnan(cand.bl_ra_nb_other) || !std::isnan(cand.bl_ra_ra_other)) {
        int nb_other_pre = (ra == nb1) ? nb2 : nb1;

        if (!std::isnan(cand.bl_ra_nb_other)) {
            bl_[ra * n_ + nb_other_pre] = bl_[nb_other_pre * n_ + ra]
                = cand.bl_ra_nb_other;
        }
        if (!std::isnan(cand.bl_ra_ra_other)) {
            int ra_nbs[3];
            int ra_nc = tree_->neighbors_of(ra, ra_nbs);
            int ra_other = -1;
            for (int i = 0; i < ra_nc; ++i)
                if (ra_nbs[i] != pu && ra_nbs[i] != nb_other_pre)
                    { ra_other = ra_nbs[i]; break; }
            if (ra_other >= 0) {
                bl_[ra * n_ + ra_other] = bl_[ra_other * n_ + ra]
                    = cand.bl_ra_ra_other;
            }
        }
    }

    // Zero the edges removed by the move.
    bl_[ra * n_ + rb]  = bl_[rb  * n_ + ra]  = 0.0;
    if (nb1 != ra && nb1 != rb)
        bl_[pu * n_ + nb1] = bl_[nb1 * n_ + pu] = 0.0;
    if (nb2 != ra && nb2 != rb)
        bl_[pu * n_ + nb2] = bl_[nb2 * n_ + pu] = 0.0;

    // calc_start moved -> full recompute.
    {
        int out[3];
        tree_->neighbors_of(0, out);
        int new_cs = out[0];
        if (new_cs != calc_start_) {
            calc_start_ = new_cs;
            _full_recompute();
            _clear_seen_radius1();
            return;
        }
    }

    // Refresh toward[], then rebuild every p_arr slot (mostly model hash-cache
    // hits + memcpy).
    _refresh_toward();
    for (int v = 0; v < n_; v++) {
        int base = v * 3;
        for (int s = 0; s < 3; s++) {
            int nb = tree_->neighbors[base + s];
            if (nb == INVALID) continue;
            model_->p_matrix(bl_[v * n_ + nb], &p_arr_[(v * 3 + s) * KK]);
        }
    }

    // Rebuild pu's toward slot. After SPR pu has neighbors {ra, rb, pv}; its
    // toward-slot CLV is the product of the other two.
    int pv = cand.prune_v;
    {
        int tw_pu = toward_[pu];
        // Compute it with _p_apply_pair on raw inputs (p_arr edge + neighbor's
        // back-CLV) to match _clv's kernel sequence exactly.
        int n_A = -1, n_B = -1;
        if      (tw_pu == pv) { n_A = ra; n_B = rb; }
        else if (tw_pu == ra) { n_A = pv; n_B = rb; }
        else if (tw_pu == rb) { n_A = pv; n_B = ra; }
        else {
            _full_recompute();
            _clear_seen_radius1();
            return;
        }

        int s_A_to_pu  = tree_->is_leaf(n_A) ? 0 : _slot_of(n_A, pu);
        int s_B_to_pu  = tree_->is_leaf(n_B) ? 0 : _slot_of(n_B, pu);
        int s_pu_to_A  = _slot_of(pu, n_A);
        int s_pu_to_B  = _slot_of(pu, n_B);
        int s_pu       = _slot_of(pu, tw_pu);

        const double* P_A   = &p_arr_[(pu  * 3 + s_pu_to_A) * KK];
        const double* P_B   = &p_arr_[(pu  * 3 + s_pu_to_B) * KK];
        const double* clv_A = tree_->is_leaf(n_A)
            ? &tip_clv_[n_A * KL]
            : _clv_slot(n_A, s_A_to_pu);
        const double* clv_B = tree_->is_leaf(n_B)
            ? &tip_clv_[n_B * KL]
            : _clv_slot(n_B, s_B_to_pu);

        // Leaves contribute log-scale 0.
        std::vector<double> ls_A_buf(L_, 0.0), ls_B_buf(L_, 0.0);
        const double* ls_A = tree_->is_leaf(n_A)
            ? ls_A_buf.data()
            : &log_scale_arr_[(n_A * 3 + s_A_to_pu) * L_];
        const double* ls_B = tree_->is_leaf(n_B)
            ? ls_B_buf.data()
            : &log_scale_arr_[(n_B * 3 + s_B_to_pu) * L_];

        double* dst    = _clv_slot(pu, s_pu);
        double* dst_ls = &log_scale_arr_[(pu * 3 + s_pu) * L_];
        _p_apply_pair(P_A, clv_A, P_B, clv_B, dst);
        for (int l = 0; l < L_; l++) dst_ls[l] = ls_A[l] + ls_B[l];
        _scale_clv_inplace(dst, dst_ls);
        dirty_[pu * 3 + s_pu] = 0;
    }

    // Propagate up to calc_start from every node whose neighbors or adjacent BL
    // changed (pu, ra, rb, nb1, nb2); the visited set dedups overlapping walks.
    std::unordered_set<int> visited;
    _propagate_up_ll(pu,  visited);
    _propagate_up_ll(ra,  visited);
    _propagate_up_ll(rb,  visited);
    _propagate_up_ll(nb1, visited);
    _propagate_up_ll(nb2, visited);

    score_ = _pseudoroot_loglik();
    _clear_seen_radius1();
}