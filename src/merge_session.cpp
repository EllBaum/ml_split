// merge_session.cpp — see merge_session.h.
#include "merge_session.h"
#include <functional>
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

std::vector<std::set<std::string>>
find_inherited_clades(const AdjTree& M, const std::set<std::string>& inh);  // fwd

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
    std::set<std::string> inh_names;                 // flatten caller grouping
    for (auto& cl : inherited)
        for (auto& nm : cl) inh_names.insert(nm);
    for (auto& nm : inh_names) remove.push_back(nm);
    std::unordered_map<std::string,int> rem_idx;
    for (int i = 0; i < (int)remove.size(); ++i) rem_idx[remove[i]] = i;

    PruneResult pr = t.pruned(remove, bl.data());

    SidePrep sp;
    sp.tree            = std::move(pr.tree);
    sp.bl              = std::move(pr.bl);
    sp.interest_anchor = pr.anchors[0];

    // Build the full adjacency, remove the interest outgroup, then auto-detect
    // the maximal inherited subtrees and capture each WITH its structure. The
    // anchor of each clade is the A' edge where it hangs (any member leaf's
    // recorded anchor — all members of a connected clade collapse to one edge).
    AdjTree full = AdjTree::from(t, bl.data());
    if (!interest.empty()) {
        std::set<std::string> is{interest};
        detach_clade(full, is);                       // suppress interest, discard
    }
    auto clades = find_inherited_clades(full, inh_names);
    for (auto& cset : clades) {
        AdjTree acopy = full;                         // interest already removed
        DetachedClade cap = detach_clade(acopy, cset);
        sp.clades.push_back(std::move(cap));
        sp.clade_anchor.push_back(pr.anchors[rem_idx[*cset.begin()]]);
    }
    return sp;
}

bool edge_exists(const AdjTree& M, int u, int v) {
    auto it = M.adj.find(u);
    if (it == M.adj.end()) return false;
    for (auto& p : it->second) if (p.first == v) return true;
    return false;
}

// Is `target` reachable from `start` without crossing `avoid`?
bool reachable(const AdjTree& M, int start, int target, int avoid) {
    std::set<int> seen{avoid};
    std::vector<int> stack{start};
    while (!stack.empty()) {
        int u = stack.back(); stack.pop_back();
        if (u == target) return true;
        if (!seen.insert(u).second) continue;
        auto it = M.adj.find(u);
        if (it == M.adj.end()) continue;
        for (auto& pr : it->second) if (pr.first != avoid) stack.push_back(pr.first);
    }
    return false;
}

// Current edge in the region of the (possibly already-consumed) anchor (p,q):
// if (p,q) still exists, that; else the edge (p, n) where n is p's neighbor on
// the path toward q (the anchor edge was split by an earlier graft / connector).
std::pair<int,int> region_edge(const AdjTree& M, int p, int q) {
    if (edge_exists(M, p, q)) return {p, q};
    for (auto& pr : M.adj.at(p))
        if (reachable(M, pr.first, q, p)) return {p, pr.first};
    throw std::runtime_error("region_edge: no path from anchor endpoint to its mate");
}

// Maximal connected all-inherited subtrees of M (with the interest outgroup
// already removed). Two inherited clades can share a host anchor edge only if
// they are connected through internal nodes — in which case they are returned
// as ONE clade with their relative structure intact. A real taxon between two
// inherited groups would split the host edge, giving them distinct anchors.
// So this grouping makes restore collision-free and structure-preserving.
std::vector<std::set<std::string>>
find_inherited_clades(const AdjTree& M, const std::set<std::string>& inh) {
    std::vector<std::set<std::string>> clades;
    int root = INVALID;
    for (auto& kv : M.leaf_name)
        if (!inh.count(kv.second) && M.adj.count(kv.first)) { root = kv.first; break; }
    if (root == INVALID) return clades;                 // no real leaf (degenerate)

    std::function<std::pair<std::set<std::string>,bool>(int,int)> dfs =
        [&](int node, int parent) -> std::pair<std::set<std::string>,bool> {
        if (M.is_leaf(node)) {
            const std::string& nm = M.leaf_name.at(node);
            return { std::set<std::string>{nm}, (bool)inh.count(nm) };
        }
        std::set<std::string> all; bool alli = true;
        std::vector<std::pair<std::set<std::string>,bool>> kids;
        for (auto& pr : M.adj.at(node)) {
            if (pr.first == parent) continue;
            auto r = dfs(pr.first, node);
            all.insert(r.first.begin(), r.first.end());
            if (!r.second) alli = false;
            kids.push_back(std::move(r));
        }
        if (alli) return { all, true };                 // bubble up to parent
        for (auto& k : kids) if (k.second) clades.push_back(k.first);  // emit maximal
        return { all, false };
    };
    int rn = M.adj.at(root)[0].first;                   // root leaf's single neighbor
    auto top = dfs(rn, root);
    if (top.second) clades.push_back(top.first);        // whole tree minus root is inh
    return clades;
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
                       int off, int cx, int cy, int newNode, int band) {
        for (size_t k = 0; k < clades.size(); ++k) {
            DetachedClade d = offset_clade(clades[k], band + (int)k * 100000);
            int p = anchors[k].first + off, q = anchors[k].second + off;
            bool is_conn = (std::min(p, q) == std::min(cx, cy) &&
                            std::max(p, q) == std::max(cx, cy));
            std::pair<int,int> e;
            if (is_conn) {
                // Connector landed on this clade's anchor edge → the edge is now
                // (cx,newNode) and (newNode,cy). Score the whole clade on each
                // sub-edge and keep the better.
                e = choose_split_subedge(M_search, d, cx, newNode, cy, full, model);
                if (!edge_exists(M, e.first, e.second))   // double-consumed: fall back
                    e = region_edge(M, p, q);
            } else if (edge_exists(M, p, q)) {
                e = {p, q};                                // anchor intact
            } else {
                // A previous inherited graft already split this anchor edge
                // (strung-along clades sharing one anchor). Attach in the region.
                e = region_edge(M, p, q);
            }
            regraft_clade_on(M, d, e.first, e.second, 0.5);
        }
    };
    restore(A.clades, A.clade_anchor, 0,
            best_eA.first, best_eA.second, newA, CLADE_BAND);
    restore(B.clades, B.clade_anchor, B_off,
            best_eB.first + B_off, best_eB.second + B_off, newB, CLADE_BAND + 50000000);

    std::vector<double> M_bl;
    Tree M_tree = M.to_tree(M_bl);

    MergeResult res;
    res.newick = M_tree.to_newick(M_bl.data());
    res.loglik = best_ll;
    return res;
}
