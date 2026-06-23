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
