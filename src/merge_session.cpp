// merge_session.cpp — see merge_session.h.
#include "merge_session.h"
#include <queue>
#include <set>
#include <unordered_map>
#include <stdexcept>

MSA slice_msa(const MSA& full, const std::vector<std::string>& names) {
    std::unordered_map<std::string,int> row_of;
    for (int i = 0; i < full.n_taxa; ++i) row_of[full.taxon_names[i]] = i;

    MSA s;
    s.seq_type         = full.seq_type;
    s.n_taxa           = (int)names.size();
    s.n_patterns       = full.n_patterns;
    s.n_sites_orig     = full.n_sites_orig;
    s.parsimony_offset = full.parsimony_offset;   // unused for likelihood
    s.weights          = full.weights;
    s.taxon_names      = names;
    s.data.resize((size_t)s.n_taxa * (size_t)s.n_patterns);

    for (int i = 0; i < s.n_taxa; ++i) {
        auto it = row_of.find(names[i]);
        if (it == row_of.end())
            throw std::runtime_error("slice_msa: taxon not in alignment: " + names[i]);
        int r = it->second;
        for (int p = 0; p < full.n_patterns; ++p)
            s.data[(size_t)i * s.n_patterns + p] = full.at(r, p);
    }
    return s;
}

std::vector<std::pair<int,int>> window_edges(const Tree& t, int a0, int a1, int n) {
    auto key = [](int u, int v) {
        return u < v ? std::make_pair(u, v) : std::make_pair(v, u);
    };
    std::vector<std::pair<int,int>> out;
    std::set<std::pair<int,int>>    seen;

    out.push_back(key(a0, a1));
    seen.insert(key(a0, a1));

    // BFS over edges: from directed edge (u->v), the next ring is the edges
    // (v,w) for v's other neighbors w. Seed both directions of the anchor.
    std::queue<std::pair<int,int>> q;
    q.push({a0, a1});
    q.push({a1, a0});

    while (!q.empty() && (int)out.size() < n) {
        auto e = q.front(); q.pop();
        int u = e.first, v = e.second;
        int nb[3]; int nc = t.neighbors_of(v, nb);
        for (int i = 0; i < nc && (int)out.size() < n; ++i) {
            int w = nb[i];
            if (w == u) continue;
            auto k = key(v, w);
            if (!seen.count(k)) {
                seen.insert(k);
                out.push_back(k);
                q.push({v, w});
            }
        }
    }
    return out;
}

// ── Merge ───────────────────────────────────────────────────────────────────
#include "likelihood_scorer.h"
#include <functional>
#include <limits>
#include <algorithm>

namespace {

struct SidePrep {
    Tree                              tree;            // A' (real leaves only)
    std::vector<double>               bl;
    std::pair<int,int>                interest_anchor; // in A' ids (seeds window)
    std::vector<DetachedClade>        clades;          // captured subtree + BLs
    std::vector<std::pair<int,int>>   clade_anchor;    // parallel, in A' ids
};

// Prune {interest} ∪ flatten(inherited); capture each inherited clade's subtree.
SidePrep prep_side(const std::string& newick,
                   const std::string& interest,
                   const std::vector<std::vector<std::string>>& inherited) {
    std::vector<double> tmp(1 << 16, 0.1);
    Tree t0 = Tree::from_newick(newick, tmp.data());
    std::vector<double> bl((size_t)t0.n_nodes * 3, 0.1);
    Tree t = Tree::from_newick(newick, bl.data(), t0.taxon_names);

    std::vector<std::string> remove;
    remove.push_back(interest);
    std::vector<int> clade_first;                  // index into remove for each clade
    for (auto& cl : inherited) {
        clade_first.push_back((int)remove.size());
        for (auto& nm : cl) remove.push_back(nm);
    }

    PruneResult pr = t.pruned(remove, bl.data());

    SidePrep sp;
    sp.tree            = std::move(pr.tree);
    sp.bl              = std::move(pr.bl);
    sp.interest_anchor = pr.anchors[0];

    for (size_t k = 0; k < inherited.size(); ++k) {
        std::set<std::string> cset(inherited[k].begin(), inherited[k].end());
        AdjTree acopy = AdjTree::from(t, bl.data());   // fresh copy per clade
        DetachedClade cap = detach_clade(acopy, cset); // capture subtree+stem
        sp.clades.push_back(std::move(cap));
        sp.clade_anchor.push_back(pr.anchors[clade_first[k]]);  // shared clade anchor
    }
    return sp;
}

bool edge_exists(const AdjTree& M, int u, int v) {
    auto it = M.adj.find(u);
    if (it == M.adj.end()) return false;
    for (auto& p : it->second) if (p.first == v) return true;
    return false;
}

// Re-token a captured clade into a unique band so it can't collide in M.
DetachedClade offset_clade(const DetachedClade& d, int off) {
    DetachedClade o;
    o.connect_token = d.connect_token + off;
    o.stem_bl       = d.stem_bl;
    o.half_a = d.half_a; o.half_b = d.half_b;
    for (auto& e : d.edges)
        o.edges.push_back({std::get<0>(e) + off, std::get<1>(e) + off, std::get<2>(e)});
    for (auto& kv : d.token_name) o.token_name[kv.first + off] = kv.second;
    return o;
}

} // namespace

std::pair<int,int> choose_split_subedge(const AdjTree& m_search,
                                        const DetachedClade& clade,
                                        int eu, int mid, int ev,
                                        const MSA& full, SubstModel& model) {
    auto score_on = [&](int a, int b) -> double {
        AdjTree cand = m_search;                 // copy (search tree only)
        regraft_clade_on(cand, clade, a, b, 0.5);
        std::vector<double> cbl;
        Tree ct = cand.to_tree(cbl);
        MSA cmsa = slice_msa(full, ct.taxon_names);
        LikelihoodScorer cs(ct, cmsa, model, cbl.data());
        return cs.score();
    };
    double ll1 = score_on(eu, mid);      // clade on (eu, mid)
    double ll2 = score_on(mid, ev);      // clade on (mid, ev)
    return (ll1 >= ll2) ? std::make_pair(eu, mid) : std::make_pair(mid, ev);
}

MergeResult run_merge(const MergeInput& in, const MSA& full, SubstModel& model,
                      int window, double connector_init) {
    SidePrep A = prep_side(in.newick_a, in.interest_a, in.inherited_a);
    SidePrep B = prep_side(in.newick_b, in.interest_b, in.inherited_b);

    MSA msaA = slice_msa(full, A.tree.taxon_names);
    MSA msaB = slice_msa(full, B.tree.taxon_names);
    LikelihoodScorer SA(A.tree, msaA, model, A.bl.data());
    LikelihoodScorer SB(B.tree, msaB, model, B.bl.data());

    auto winA = window_edges(A.tree, A.interest_anchor.first,  A.interest_anchor.second,  window);
    auto winB = window_edges(B.tree, B.interest_anchor.first,  B.interest_anchor.second,  window);

    // Precompute B-side boundary CLVs once (independent of the A loop).
    struct BndB { std::vector<double> nb, nbl, ro, rol; double bl; };
    std::vector<BndB> bcache(winB.size());
    for (size_t j = 0; j < winB.size(); ++j) {
        SB.clv(winB[j].first,  winB[j].second, bcache[j].nb, bcache[j].nbl);
        SB.clv(winB[j].second, winB[j].first,  bcache[j].ro, bcache[j].rol);
        bcache[j].bl = SB.branch_length(winB[j].first, winB[j].second);
    }

    double best_ll = -std::numeric_limits<double>::infinity();
    std::pair<int,int> best_eA{INVALID,INVALID}, best_eB{INVALID,INVALID};
    double best_bl5[5] = {0,0,0,0,0};

    std::vector<double> pv, pvl, rb, rbl;
    for (auto& eA : winA) {
        SA.clv(eA.first,  eA.second, pv, pvl);
        SA.clv(eA.second, eA.first,  rb, rbl);
        double blA = SA.branch_length(eA.first, eA.second);
        for (size_t j = 0; j < winB.size(); ++j) {
            double init5[5] = {connector_init, 0.5 * blA, 0.5 * blA,
                               0.5 * bcache[j].bl, 0.5 * bcache[j].bl};
            double bl5[5];
            double ll = SA.optimize_five_for_merge(
                pv.data(), pvl.data(), rb.data(), rbl.data(),
                bcache[j].nb.data(), bcache[j].nbl.data(),
                bcache[j].ro.data(), bcache[j].rol.data(),
                init5, bl5);
            if (ll > best_ll) {
                best_ll = ll; best_eA = eA; best_eB = winB[j];
                std::copy(bl5, bl5 + 5, best_bl5);
            }
        }
    }

    // ── Assemble the winner in AdjTree token space ───────────────────────────
    const int B_off = A.tree.n_nodes;
    AdjTree M;
    auto add_side = [&](const Tree& t, const std::vector<double>& bl, int off) {
        for (int u = 0; u < t.n_nodes; ++u) {
            if (t.is_leaf(u)) M.leaf_name[u + off] = t.taxon_names[u];
            int nb3[3]; int nc = t.neighbors_of(u, nb3);
            for (int i = 0; i < nc; ++i) {
                int v = nb3[i];
                if (u < v)
                    M.connect(u + off, v + off,
                              Tree::bl_get(t.neighbors, bl.data(), u, v));
            }
        }
    };
    add_side(A.tree, A.bl, 0);
    add_side(B.tree, B.bl, B_off);
    M.fresh = A.tree.n_nodes + B.tree.n_nodes + 16;

    M.disconnect(best_eA.first, best_eA.second);
    int newA = M.new_internal();
    M.connect(best_eA.first,  newA, best_bl5[1]);
    M.connect(newA, best_eA.second, best_bl5[2]);

    M.disconnect(best_eB.first + B_off, best_eB.second + B_off);
    int newB = M.new_internal();
    M.connect(best_eB.first + B_off,  newB, best_bl5[3]);
    M.connect(newB, best_eB.second + B_off, best_bl5[4]);

    M.connect(newA, newB, best_bl5[0]);

    // Snapshot the search tree (realA ∪ realB, no inherited) for split-edge
    // likelihood scoring below.
    AdjTree M_search = M;

    // ── Restore inherited clades ─────────────────────────────────────────────
    const int CLADE_BAND = A.tree.n_nodes + B.tree.n_nodes + 1000;
    auto restore = [&](std::vector<DetachedClade>& clades,
                       std::vector<std::pair<int,int>>& anchors,
                       int off, int newNode, int band) {
        for (size_t k = 0; k < clades.size(); ++k) {
            DetachedClade d = offset_clade(clades[k], band + (int)k * 100000);
            int p = anchors[k].first + off, q = anchors[k].second + off;
            if (edge_exists(M, p, q)) {
                regraft_clade_on(M, d, p, q, 0.5);          // non-split
            } else {
                // Split case: connector landed on this clade's anchor edge, now
                // two sub-edges (p,newNode) and (newNode,q). Score the whole
                // clade on each and keep the better.
                auto e = choose_split_subedge(M_search, d, p, newNode, q, full, model);
                regraft_clade_on(M, d, e.first, e.second, 0.5);
            }
        }
    };
    restore(A.clades, A.clade_anchor, 0,     newA, CLADE_BAND);
    restore(B.clades, B.clade_anchor, B_off, newB, CLADE_BAND + 50000000);

    std::vector<double> M_bl;
    Tree M_tree = M.to_tree(M_bl);

    MergeResult res;
    res.newick = M_tree.to_newick(M_bl.data());
    res.loglik = best_ll;
    return res;
}
