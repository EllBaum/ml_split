// merge_prep.cpp — see merge_prep.h.
#include "merge_prep.h"
#include <algorithm>
#include <functional>
#include <stdexcept>

// ── AdjTree ───────────────────────────────────────────────────────────────────

AdjTree AdjTree::from(const Tree& t, const double* bl) {
    AdjTree a;
    for (int u = 0; u < t.n_nodes; ++u) {
        for (int k = 0; k < 3; ++k) {
            int v = t.neighbors[u * 3 + k];
            if (v != INVALID && u < v) {
                double w = Tree::bl_get(t.neighbors, bl, u, v);
                a.adj[u].push_back({v, w});
                a.adj[v].push_back({u, w});
            }
        }
        if (t.is_leaf(u)) a.leaf_name[u] = t.taxon_names[u];
    }
    a.fresh = t.n_nodes;
    return a;
}

double AdjTree::get_bl(int u, int v) const {
    for (auto& p : adj.at(u)) if (p.first == v) return p.second;
    throw std::runtime_error("AdjTree::get_bl: no edge");
}

void AdjTree::connect(int u, int v, double w) {
    adj[u].push_back({v, w});
    adj[v].push_back({u, w});
}

void AdjTree::disconnect(int u, int v) {
    auto rm = [&](int a, int b) {
        auto& vec = adj.at(a);
        vec.erase(std::remove_if(vec.begin(), vec.end(),
                  [&](auto& p) { return p.first == b; }), vec.end());
    };
    rm(u, v); rm(v, u);
}

Tree AdjTree::to_tree(std::vector<double>& bl_out) const {
    std::vector<std::tuple<int,int,double>> edges;
    std::unordered_map<int,std::string>     leaf_tok;
    std::set<std::pair<int,int>>            done;
    for (auto& kv : adj) {
        int u = kv.first;
        if (is_leaf(u)) leaf_tok[u] = leaf_name.at(u);
        for (auto& p : kv.second) {
            auto key = std::minmax(u, p.first);
            if (done.count({key.first, key.second})) continue;
            done.insert({key.first, key.second});
            edges.push_back({u, p.first, p.second});
        }
    }
    return Tree::assemble(edges, leaf_tok, bl_out);
}

// ── stem / detach / regraft ────────────────────────────────────────────────────

std::pair<int,int> find_stem(const AdjTree& a, const std::set<std::string>& clade) {
    int ref = -1;
    for (auto& kv : a.leaf_name)
        if (a.adj.count(kv.first) && !clade.count(kv.second)) { ref = kv.first; break; }
    if (ref < 0) throw std::runtime_error("find_stem: no external reference leaf");

    std::pair<int,int> stem{INVALID, INVALID};
    std::function<std::set<std::string>(int,int)> dfs =
        [&](int node, int parent) -> std::set<std::string> {
            std::set<std::string> below;
            if (a.is_leaf(node)) below.insert(a.leaf_name.at(node));
            for (auto& p : a.adj.at(node))
                if (p.first != parent) {
                    auto sub = dfs(p.first, node);
                    below.insert(sub.begin(), sub.end());
                }
            if (below == clade && stem.first == INVALID) stem = {parent, node};  // (P,c)
            return below;
        };
    dfs(ref, INVALID);
    if (stem.first == INVALID)
        throw std::runtime_error("find_stem: leaves do not form a clade");
    return stem;
}

DetachedClade detach_clade(AdjTree& a, const std::set<std::string>& clade) {
    auto stem = find_stem(a, clade);
    int P = stem.first, c = stem.second;

    DetachedClade d;
    d.stem_bl       = a.get_bl(P, c);
    d.connect_token = c;

    std::function<void(int,int)> grab = [&](int node, int parent) {
        if (a.is_leaf(node)) d.token_name[node] = a.leaf_name.at(node);
        for (auto& p : a.adj.at(node))
            if (p.first != parent) {
                d.edges.push_back({node, p.first, p.second});
                grab(p.first, node);
            }
    };
    grab(c, P);

    a.disconnect(P, c);
    std::function<void(int,int)> kill = [&](int node, int parent) {
        std::vector<int> kids;
        for (auto& p : a.adj.at(node)) if (p.first != parent) kids.push_back(p.first);
        for (int kch : kids) kill(kch, node);
        a.erase_node(node);
    };
    kill(c, P);

    if (a.degree(P) != 2)
        throw std::runtime_error("detach_clade: host node not degree 2 after detach");
    int X = a.adj.at(P)[0].first, Y = a.adj.at(P)[1].first;
    d.half_a = a.get_bl(P, X);
    d.half_b = a.get_bl(P, Y);
    a.disconnect(P, X); a.disconnect(P, Y); a.erase_node(P);
    a.connect(X, Y, d.half_a + d.half_b);
    d.anchor = {X, Y};
    return d;
}

static void readd_clade_nodes(AdjTree& a, const DetachedClade& d) {
    for (auto& kv : d.token_name) a.leaf_name[kv.first] = kv.second;
    std::set<std::pair<int,int>> seen;
    for (auto& e : d.edges) {
        int u = std::get<0>(e), v = std::get<1>(e);
        double w = std::get<2>(e);
        auto key = std::minmax(u, v);
        if (seen.count({key.first, key.second})) continue;
        seen.insert({key.first, key.second});
        a.connect(u, v, w);
    }
}

void regraft_clade(AdjTree& a, const DetachedClade& d) {
    int X = d.anchor.first, Y = d.anchor.second;
    a.disconnect(X, Y);
    int P = a.new_internal();
    a.connect(X, P, d.half_a);
    a.connect(P, Y, d.half_b);
    readd_clade_nodes(a, d);
    a.connect(P, d.connect_token, d.stem_bl);
}

void regraft_clade_on(AdjTree& a, const DetachedClade& d,
                      int edge_u, int edge_v, double frac) {
    double w = a.get_bl(edge_u, edge_v);
    a.disconnect(edge_u, edge_v);
    int P = a.new_internal();
    a.connect(edge_u, P, w * frac);
    a.connect(P, edge_v, w * (1.0 - frac));
    readd_clade_nodes(a, d);
    a.connect(P, d.connect_token, d.stem_bl);
}
