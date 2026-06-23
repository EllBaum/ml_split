// tree.cpp — Implementation of Tree (edge-free version)

#include "tree.h"
#include <cstdlib>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <stack>
#include <queue>
#include <unordered_map>
#include <functional>
#include <cstring>
#include <array>
#include <tuple>
#include <set>

// ── Constructor ───────────────────────────────────────────────────────────────

Tree::Tree(int n_taxa, const std::vector<std::string>& taxon_names)
    : n_taxa(n_taxa),
      n_nodes(2 * n_taxa - 2),
      n_edges(2 * n_taxa - 3),
      taxon_names(taxon_names),
      neighbors(n_nodes * 3, INVALID)
{
    assert(n_taxa >= 3);
    assert(static_cast<int>(taxon_names.size()) == n_taxa);
}

// ── Private neighbor manipulation ────────────────────────────────────────────

void Tree::add_neighbor(int node, int neighbor) {
    int base = node * 3;
    for (int k = 0; k < 3; ++k) {
        if (neighbors[base + k] == INVALID) {
            neighbors[base + k] = neighbor;
            return;
        }
    }
    throw std::runtime_error("Node " + std::to_string(node) +
                             " already has 3 neighbors");
}

void Tree::remove_neighbor(int node, int neighbor) {
    int base = node * 3;
    for (int k = 0; k < 3; ++k) {
        if (neighbors[base + k] == neighbor) {
            neighbors[base + k] = INVALID;
            return;
        }
    }
    throw std::runtime_error("Node " + std::to_string(node) +
                             " has no neighbor " + std::to_string(neighbor));
}

void Tree::attach(int u, int v) {
    add_neighbor(u, v);
    add_neighbor(v, u);
}

void Tree::detach(int u, int v) {
    remove_neighbor(u, v);
    remove_neighbor(v, u);
}

// ── Branch length helpers ─────────────────────────────────────────────────────

double Tree::bl_get(const std::vector<int32_t>& neighbors,
                    const double* bl, int u, int v) {
    int base = u * 3;
    for (int k = 0; k < 3; ++k) {
        if (neighbors[base + k] == v)
            return bl[base + k];
    }
    throw std::runtime_error("bl_get: no edge between " +
                             std::to_string(u) + " and " + std::to_string(v));
}

void Tree::bl_set(const std::vector<int32_t>& neighbors,
                  double* bl, int u, int v, double value) {
    bool set_u = false, set_v = false;
    int base_u = u * 3, base_v = v * 3;
    for (int k = 0; k < 3; ++k) {
        if (neighbors[base_u + k] == v) { bl[base_u + k] = value; set_u = true; }
        if (neighbors[base_v + k] == u) { bl[base_v + k] = value; set_v = true; }
    }
    if (!set_u || !set_v)
        throw std::runtime_error("bl_set: no edge between " +
                                 std::to_string(u) + " and " + std::to_string(v));
}

std::vector<double> Tree::make_bl(int n_nodes, double default_val) {
    return std::vector<double>(n_nodes * 3, default_val);
}

// ── Query helpers ─────────────────────────────────────────────────────────────

int Tree::degree(int node) const {
    int base = node * 3;
    int d = 0;
    for (int k = 0; k < 3; ++k)
        if (neighbors[base + k] != INVALID) ++d;
    return d;
}

bool Tree::are_connected(int u, int v) const {
    int base = u * 3;
    for (int k = 0; k < 3; ++k)
        if (neighbors[base + k] == v) return true;
    return false;
}

int Tree::neighbors_of(int node, int out[3]) const {
    int base  = node * 3;
    int count = 0;
    for (int k = 0; k < 3; ++k)
        if (neighbors[base + k] != INVALID)
            out[count++] = neighbors[base + k];
    return count;
}

int Tree::neighbors_except(int node, int exclude, int out[2]) const {
    int base  = node * 3;
    int count = 0;
    for (int k = 0; k < 3; ++k) {
        int nb = neighbors[base + k];
        if (nb != INVALID && nb != exclude)
            out[count++] = nb;
    }
    return count;
}

int Tree::other_neighbor(int node, int excl1, int excl2) const {
    int base = node * 3;
    for (int k = 0; k < 3; ++k) {
        int nb = neighbors[base + k];
        if (nb != INVALID && nb != excl1 && nb != excl2)
            return nb;
    }
    return INVALID;
}

// ── Traversal ─────────────────────────────────────────────────────────────────

std::vector<int> Tree::postorder(int start, int exclude) const {
    std::vector<int> result;
    result.reserve(n_nodes);

    // TraversalNode: node + direction we came from
    struct TraversalNode { int node, came_from; };
    std::stack<TraversalNode> traversal_stack;
    traversal_stack.push({start, exclude});

    while (!traversal_stack.empty()) {
        TraversalNode top = traversal_stack.top();
        traversal_stack.pop();
        result.push_back(top.node);

        int base = top.node * 3;
        for (int k = 0; k < 3; ++k) {
            int nb = neighbors[base + k];
            if (nb != INVALID && nb != top.came_from)
                traversal_stack.push({nb, top.node});
        }
    }

    std::reverse(result.begin(), result.end());
    return result;
}

// ── Random tree ───────────────────────────────────────────────────────────────

Tree Tree::random_tree(const std::vector<std::string>& taxa,
                       std::mt19937& rng) {
    int n = static_cast<int>(taxa.size());
    if (n < 3) throw std::runtime_error("Need >= 3 taxa");

    Tree t(n, taxa);

    int first_internal = n;
    t.attach(0, first_internal);
    t.attach(1, first_internal);
    t.attach(2, first_internal);

    struct Edge { int u, v; };
    std::vector<Edge> edges = {{0, first_internal},
                               {1, first_internal},
                               {2, first_internal}};
    int next_internal = first_internal + 1;

    for (int sp = 3; sp < n; ++sp) {
        std::uniform_int_distribution<int> dist(0,
            static_cast<int>(edges.size()) - 1);
        int idx = dist(rng);
        Edge e  = edges[idx];
        edges.erase(edges.begin() + idx);

        t.detach(e.u, e.v);

        int ni = next_internal++;
        t.attach(e.u, ni);
        t.attach(e.v, ni);
        t.attach(sp,  ni);

        edges.push_back({e.u, ni});
        edges.push_back({e.v, ni});
        edges.push_back({sp,  ni});
    }

    return t;
}

// ── SPR enumeration ───────────────────────────────────────────────────────────

std::vector<std::pair<int,int>> Tree::spr_prune_points() const {
    std::vector<std::pair<int,int>> result;
    for (int pu = n_taxa; pu < n_nodes; ++pu) {
        int base = pu * 3;
        for (int k = 0; k < 3; ++k) {
            int pv = neighbors[base + k];
            if (pv == INVALID) continue;
            // only include if remaining tree has at least one valid regraft
            // candidate: at least one of pu's other neighbors must be internal
            int others[2];
            int nc = neighbors_except(pu, pv, others);
            if (nc == 2) {
                bool has_internal_other = !is_leaf(others[0]) ||
                                          !is_leaf(others[1]);
                if (has_internal_other)
                    result.push_back({pu, pv});
            }
        }
    }
    return result;
}

std::vector<std::pair<int,int>> Tree::spr_candidates(int prune_u, int prune_v,
                                                       int radius) const {
    std::vector<std::pair<int,int>> candidates;

    struct DFSNode { int node, came_from, depth; };
    std::stack<DFSNode> dfs_stack;
    dfs_stack.push({prune_u, prune_v, 0});

    while (!dfs_stack.empty()) {
        DFSNode top = dfs_stack.top();
        dfs_stack.pop();
        int base = top.node * 3;
        for (int k = 0; k < 3; ++k) {
            int nb = neighbors[base + k];
            if (nb == INVALID || nb == top.came_from) continue;
            if (top.depth >= 1)
                candidates.push_back({top.node, nb});
            if (top.depth < radius && !is_leaf(nb))
                dfs_stack.push({nb, top.node, top.depth + 1});
        }
    }
    return candidates;
}

// ── SPR move ──────────────────────────────────────────────────────────────────

Tree Tree::spr_move(int prune_u, int prune_v,
                    int regraft_a, int regraft_b,
                    bool in_place) {
    assert(!is_leaf(prune_u));

    if (!in_place) {
        Tree t = *this;  // deep copy via std::vector copy constructor
        t.spr_move(prune_u, prune_v, regraft_a, regraft_b, true);
        return t;
    }

    int others[2];
    [[maybe_unused]] int nc = neighbors_except(prune_u, prune_v, others);
    assert(nc == 2);
    int nb1 = others[0], nb2 = others[1];

    // Phase 1: detach prune_u from everything
    detach(prune_u, prune_v);
    detach(prune_u, nb1);
    detach(prune_u, nb2);

    // bypass prune_u: connect nb1 directly to nb2
    attach(nb1, nb2);

    // Phase 2: split regraft edge and reinsert prune_u
    detach(regraft_a, regraft_b);
    attach(regraft_a, prune_u);
    attach(regraft_b, prune_u);
    attach(prune_u,   prune_v);

    return Tree{};  // in_place=true returns empty tree (ignored by caller)
}

// ── NNI move ──────────────────────────────────────────────────────────────────

Tree Tree::nni_move(int edge_u, int edge_v, int swap_which, bool in_place) {
    assert(!is_leaf(edge_u) && !is_leaf(edge_v));

    if (!in_place) {
        Tree t = *this;
        t.nni_move(edge_u, edge_v, swap_which, true);
        return t;
    }

    int u_nbs[2], v_nbs[2];
    neighbors_except(edge_u, edge_v, u_nbs);
    neighbors_except(edge_v, edge_u, v_nbs);

    int b = u_nbs[1];
    int c = v_nbs[swap_which];

    detach(edge_u, b);
    detach(edge_v, c);
    attach(edge_u, c);
    attach(edge_v, b);

    return Tree{};
}

// ── Newick I/O ────────────────────────────────────────────────────────────────

std::string Tree::subtree_newick(int node, int exclude,
                                 const double* bl) const {
    std::ostringstream ss;

    if (is_leaf(node)) {
        ss << taxon_names[node];
    } else {
        int nbs[2];
        [[maybe_unused]] int nc = neighbors_except(node, exclude, nbs);
        assert(nc == 2);
        ss << "("
           << subtree_newick(nbs[0], node, bl)
           << ","
           << subtree_newick(nbs[1], node, bl)
           << ")";
    }

    if (exclude != INVALID && bl != nullptr) {
        double val = bl_get(neighbors, bl, node, exclude);
        ss << ":" << std::setprecision(15) << val;
    }

    return ss.str();
}

std::string Tree::to_newick(const double* bl) const {
    int start = n_taxa;
    int nbs[3];
    [[maybe_unused]] int nc = neighbors_of(start, nbs);
    assert(nc == 3);

    std::ostringstream ss;
    ss << "(";
    for (int i = 0; i < 3; ++i) {
        if (i > 0) ss << ",";
        ss << subtree_newick(nbs[i], start, bl);
    }
    ss << ");";
    return ss.str();
}

// ── Newick parsing ────────────────────────────────────────────────────────────

static std::vector<std::string> extract_taxa_from_newick(const std::string& newick) {
    std::vector<std::string> names;
    std::string s = newick;
    while (!s.empty() && (s.back() == ';' || s.back() == ' '))
        s.pop_back();

    int pos = 0;
    int len = static_cast<int>(s.size());
    while (pos < len) {
        char c = s[pos];
        if (c == '(' || c == ',') {
            ++pos;
        } else if (c == ')') {
            ++pos;
            if (pos < len && s[pos] == ':') {
                ++pos;
                while (pos < len && s[pos] != ',' && s[pos] != ')') ++pos;
            }
        } else if (c == ':') {
            ++pos;
            while (pos < len && s[pos] != ',' && s[pos] != ')') ++pos;
        } else {
            int start = pos;
            while (pos < len && s[pos] != ':' && s[pos] != ',' && s[pos] != ')')
                ++pos;
            std::string name = s.substr(start, pos - start);
            while (!name.empty() && name.back()  == ' ') name.pop_back();
            while (!name.empty() && name.front() == ' ') name.erase(name.begin());
            if (!name.empty()) names.push_back(name);
        }
    }
    return names;
}

Tree Tree::from_newick(const std::string& newick,
                       double* bl_out,
                       const std::vector<std::string>& taxa_order) {
    std::vector<std::string> taxa = taxa_order.empty()
        ? extract_taxa_from_newick(newick)
        : taxa_order;

    int n = static_cast<int>(taxa.size());
    Tree t(n, taxa);

    // Initialize bl_out to 0.1
    std::fill(bl_out, bl_out + t.n_nodes * 3, 0.1);

    std::unordered_map<std::string, int> name_to_id;
    for (int i = 0; i < n; ++i)
        name_to_id[taxa[i]] = i;

    std::string s = newick;
    while (!s.empty() && (s.back() == ';' || s.back() == ' ' || s.back() == '\n'))
        s.pop_back();
    while (!s.empty() && (s.front() == ' ' || s.front() == '\n'))
        s.erase(s.begin());

    int next_internal = n;
    int pos = 0;

    struct ParseResult { int node_id; double bl; };

    std::function<ParseResult()> parse = [&]() -> ParseResult {
        if (s[pos] == '(') {
            ++pos;
            int node_id = next_internal++;

            while (true) {
                auto sub = parse();
                t.attach(node_id, sub.node_id);
                // write branch length into bl_out immediately after attach
                Tree::bl_set(t.neighbors, bl_out, node_id, sub.node_id, sub.bl);
                if (s[pos] == ',') {
                    ++pos;
                } else {
                    break;
                }
            }

            assert(s[pos] == ')');
            ++pos;

            double bl = 0.1;
            if (pos < static_cast<int>(s.size()) && s[pos] == ':') {
                ++pos;
                int start = pos;
                while (pos < static_cast<int>(s.size()) &&
                       (s[pos] == '.' || s[pos] == '-' || s[pos] == 'e' ||
                        s[pos] == 'E' || s[pos] == '+' ||
                        (s[pos] >= '0' && s[pos] <= '9')))
                    ++pos;
                bl = std::stod(s.substr(start, pos - start));
            }

            return {node_id, bl};
        } else {
            int start = pos;
            while (pos < static_cast<int>(s.size()) &&
                   s[pos] != ',' && s[pos] != ')' && s[pos] != ':')
                ++pos;
            std::string name = s.substr(start, pos - start);
            while (!name.empty() && name.back()  == ' ') name.pop_back();
            while (!name.empty() && name.front() == ' ') name.erase(name.begin());

            double bl = 0.1;
            if (pos < static_cast<int>(s.size()) && s[pos] == ':') {
                ++pos;
                int bstart = pos;
                while (pos < static_cast<int>(s.size()) &&
                       (s[pos] == '.' || s[pos] == '-' || s[pos] == 'e' ||
                        s[pos] == 'E' || s[pos] == '+' ||
                        (s[pos] >= '0' && s[pos] <= '9')))
                    ++pos;
                bl = std::stod(s.substr(bstart, pos - bstart));
            }

            auto it = name_to_id.find(name);
            if (it == name_to_id.end())
                throw std::runtime_error("Unknown taxon '" + name + "' in Newick");

            return {it->second, bl};
        }
    };

    parse();
    assert(next_internal == t.n_nodes);
    return t;
}

// ── Validation ────────────────────────────────────────────────────────────────

bool Tree::validate() const {
    bool ok = true;

    for (int i = 0; i < n_taxa; ++i) {
        if (degree(i) != 1) {
            std::cerr << "Leaf " << i << " (" << taxon_names[i]
                      << ") has degree " << degree(i) << "\n";
            ok = false;
        }
    }

    for (int i = n_taxa; i < n_nodes; ++i) {
        if (degree(i) != 3) {
            std::cerr << "Internal node " << i
                      << " has degree " << degree(i) << "\n";
            ok = false;
        }
    }

    if (!ok) return false;

    // Connectivity via BFS (Breadth-First Search)
    std::vector<bool> visited(n_nodes, false);
    std::queue<int> q;
    q.push(0);
    visited[0] = true;
    int count = 0;
    while (!q.empty()) {
        int cur = q.front(); q.pop();
        ++count;
        int base = cur * 3;
        for (int k = 0; k < 3; ++k) {
            int nb = neighbors[base + k];
            if (nb != INVALID && !visited[nb]) {
                visited[nb] = true;
                q.push(nb);
            }
        }
    }
    if (count != n_nodes) {
        std::cerr << "Disconnected: visited " << count
                  << " / " << n_nodes << "\n";
        return false;
    }

    // Symmetry
    for (int u = 0; u < n_nodes; ++u) {
        int base = u * 3;
        for (int k = 0; k < 3; ++k) {
            int nb = neighbors[base + k];
            if (nb == INVALID) continue;
            if (!are_connected(nb, u)) {
                std::cerr << "Asymmetric: " << u << " -> " << nb << "\n";
                return false;
            }
        }
    }

    return true;
}

// ── Debug ─────────────────────────────────────────────────────────────────────

void Tree::print_adjacency() const {
    for (int node = 0; node < n_nodes; ++node) {
        std::string label = is_leaf(node) ? taxon_names[node]
                                          : "N" + std::to_string(node);
        std::cerr << "  " << label << "(" << node << "): neighbors=[";
        int base = node * 3;
        bool first = true;
        for (int k = 0; k < 3; ++k) {
            if (neighbors[base + k] != INVALID) {
                if (!first) std::cerr << ",";
                std::cerr << neighbors[base + k];
                first = false;
            }
        }
        std::cerr << "]\n";
    }
}

// ── Leaf pruning + anchor recording ───────────────────────────────────────────

PruneResult Tree::pruned(const std::vector<std::string>& remove_leaf_names,
                         const double* bl) const {
    // Resolve names → original leaf ids.
    std::unordered_map<std::string,int> name_to_id;
    for (int i = 0; i < n_taxa; ++i) name_to_id[taxon_names[i]] = i;

    std::vector<int> remove_ids;
    remove_ids.reserve(remove_leaf_names.size());
    for (const auto& nm : remove_leaf_names) {
        auto it = name_to_id.find(nm);
        if (it == name_to_id.end())
            throw std::runtime_error("pruned: '" + nm + "' is not a leaf of this tree");
        remove_ids.push_back(it->second);   // < n_taxa by construction
    }

    // Mutable adjacency + bl over original node ids (≤3 slots each).
    std::vector<std::array<int,3>>    adj(n_nodes);
    std::vector<std::array<double,3>> adjbl(n_nodes);
    for (int u = 0; u < n_nodes; ++u)
        for (int k = 0; k < 3; ++k) {
            adj[u][k]   = neighbors[u * 3 + k];
            adjbl[u][k] = bl ? bl[u * 3 + k] : 0.1;
        }
    std::vector<char> alive(n_nodes, 1);

    auto edge_key = [](int a, int b) -> int64_t {
        if (a > b) std::swap(a, b);
        return (static_cast<int64_t>(a) << 32) | static_cast<uint32_t>(b);
    };
    auto deg_of = [&](int u) {
        int d = 0; for (int k = 0; k < 3; ++k) if (adj[u][k] != INVALID) ++d; return d;
    };
    auto get_bl = [&](int u, int v) -> double {
        for (int k = 0; k < 3; ++k) if (adj[u][k] == v) return adjbl[u][k];
        throw std::runtime_error("pruned: get_bl missing edge");
    };

    // Union-find over edge ids; ep[find(id)] tracks current live endpoints.
    std::unordered_map<int64_t,int> eid;          // live edge key → edge id
    std::vector<int>                uf;            // edge id → parent
    std::vector<std::pair<int,int>> ep;           // edge id → endpoints (live)
    std::function<int(int)> find = [&](int x) {
        while (uf[x] != x) { uf[x] = uf[uf[x]]; x = uf[x]; }
        return x;
    };
    auto new_edge = [&](int u, int v) -> int {
        int id = static_cast<int>(uf.size());
        uf.push_back(id); ep.push_back({u, v});
        return id;
    };
    auto disconnect = [&](int u, int v) -> int {       // returns the dying edge id
        int64_t key = edge_key(u, v);
        int id = eid.at(key);
        eid.erase(key);
        for (int k = 0; k < 3; ++k) if (adj[u][k] == v) { adj[u][k] = INVALID; adjbl[u][k] = 0; }
        for (int k = 0; k < 3; ++k) if (adj[v][k] == u) { adj[v][k] = INVALID; adjbl[v][k] = 0; }
        return id;
    };
    auto connect_id = [&](int u, int v, double w, int id) {
        for (int k = 0; k < 3; ++k) if (adj[u][k] == INVALID) { adj[u][k] = v; adjbl[u][k] = w; break; }
        for (int k = 0; k < 3; ++k) if (adj[v][k] == INVALID) { adj[v][k] = u; adjbl[v][k] = w; break; }
        eid[edge_key(u, v)] = id;
        ep[id] = {u, v};
    };

    // Seed edge ids for every original live edge.
    for (int u = 0; u < n_nodes; ++u)
        for (int k = 0; k < 3; ++k) {
            int v = adj[u][k];
            if (v != INVALID && u < v) eid[edge_key(u, v)] = new_edge(u, v);
        }

    // Reduce a node that just lost a neighbor. `incoming` is the dead edge id
    // (removed pendant / stub) that must follow to the surviving merged edge.
    std::function<int(int,int)> reduce = [&](int u, int incoming) -> int {
        int d = deg_of(u);
        if (d == 2) {
            int X = INVALID, Y = INVALID;
            for (int k = 0; k < 3; ++k)
                if (adj[u][k] != INVALID) { (X == INVALID ? X : Y) = adj[u][k]; }
            double bX = get_bl(u, X), bY = get_bl(u, Y);
            int idX = disconnect(u, X);
            int idY = disconnect(u, Y);
            alive[u] = 0;
            int rep = find(incoming);
            int rX = find(idX); if (rX != rep) uf[rX] = rep;
            int rY = find(idY); if (rY != rep) uf[rY] = rep;
            connect_id(X, Y, bX + bY, rep);     // ep[rep] = (X,Y)
            return rep;
        } else if (d == 1) {                    // stub cascade (not hit by binary one-at-a-time)
            int Z = INVALID;
            for (int k = 0; k < 3; ++k) if (adj[u][k] != INVALID) Z = adj[u][k];
            int idZ = disconnect(u, Z);
            alive[u] = 0;
            int rep = find(incoming);
            int rZ = find(idZ); if (rZ != rep) uf[rZ] = rep;
            return reduce(Z, rep);
        }
        throw std::runtime_error("pruned: unexpected degree " + std::to_string(d) +
                                 " during reduction");
    };

    // Remove each leaf; record the anchor edge id it collapses onto.
    std::vector<int> anchor_id(remove_ids.size());
    for (size_t i = 0; i < remove_ids.size(); ++i) {
        int L = remove_ids[i];
        if (!alive[L]) throw std::runtime_error("pruned: leaf removed twice");
        if (deg_of(L) != 1) throw std::runtime_error("pruned: target is not a pendant leaf");
        int P = INVALID;
        for (int k = 0; k < 3; ++k) if (adj[L][k] != INVALID) P = adj[L][k];
        int idLP = disconnect(L, P);
        alive[L] = 0;
        anchor_id[i] = reduce(P, idLP);
    }

    // Renumber survivors: leaves first (ascending old id), then internal nodes.
    std::vector<int> old2new(n_nodes, INVALID);
    std::vector<std::string> new_names;
    int k = 0;
    for (int u = 0; u < n_taxa; ++u)
        if (alive[u]) { old2new[u] = k++; new_names.push_back(taxon_names[u]); }
    if (k < 3)
        throw std::runtime_error("pruned: fewer than 3 leaves remain (" + std::to_string(k) + ")");
    int next_internal = k;
    for (int u = n_taxa; u < n_nodes; ++u)
        if (alive[u]) old2new[u] = next_internal++;

    if (std::getenv("PRUNE_DEBUG")) {
        std::cerr << "[pruned] k_leaves=" << k << " survivors_internal="
                  << (next_internal - k) << " expected=" << (k - 2) << "\n";
        for (int u = 0; u < n_nodes; ++u) if (alive[u]) {
            std::cerr << "   alive " << u << " deg=" << deg_of(u) << " nbrs=[";
            for (int kk = 0; kk < 3; ++kk) if (adj[u][kk] != INVALID) std::cerr << adj[u][kk] << " ";
            std::cerr << "]\n";
        }
    }

    PruneResult res;
    res.tree = Tree(k, new_names);
    if (next_internal != res.tree.n_nodes)
        throw std::runtime_error("pruned: survivor count mismatch (got " +
                                 std::to_string(next_internal) + ", expected " +
                                 std::to_string(res.tree.n_nodes) + ")");
    res.bl = Tree::make_bl(res.tree.n_nodes, 0.1);

    for (int u = 0; u < n_nodes; ++u) {
        if (!alive[u]) continue;
        for (int kk = 0; kk < 3; ++kk) {
            int v = adj[u][kk];
            if (v != INVALID && u < v) {
                int nu = old2new[u], nv = old2new[v];
                res.tree.attach(nu, nv);
                Tree::bl_set(res.tree.neighbors, res.bl.data(), nu, nv, adjbl[u][kk]);
            }
        }
    }

    res.removed_names = remove_leaf_names;
    res.anchors.resize(remove_ids.size());
    for (size_t i = 0; i < remove_ids.size(); ++i) {
        std::pair<int,int> e = ep[find(anchor_id[i])];
        res.anchors[i] = { old2new[e.first], old2new[e.second] };
    }
    return res;
}

// ── Assemble a renumbered tree from an explicit edge list ─────────────────────

Tree Tree::assemble(const std::vector<std::tuple<int,int,double>>& edges,
                    const std::unordered_map<int,std::string>& leaf_tokens,
                    std::vector<double>& bl_out) {
    // Leaf ids: sorted by name. Internal ids: follow, in first-seen token order.
    std::vector<std::string> names;
    names.reserve(leaf_tokens.size());
    for (const auto& kv : leaf_tokens) names.push_back(kv.second);
    std::sort(names.begin(), names.end());
    std::unordered_map<std::string,int> name2new;
    for (int i = 0; i < (int)names.size(); ++i) name2new[names[i]] = i;
    int k = (int)names.size();

    std::unordered_map<int,int> tok2new;
    for (const auto& kv : leaf_tokens) tok2new[kv.first] = name2new[kv.second];
    int next_internal = k;
    auto map_tok = [&](int tok) -> int {
        auto it = tok2new.find(tok);
        if (it != tok2new.end()) return it->second;
        int id = next_internal++;
        tok2new[tok] = id;
        return id;
    };
    // First pass: assign internal ids deterministically (first-seen across edges).
    for (const auto& e : edges) { map_tok(std::get<0>(e)); map_tok(std::get<1>(e)); }

    Tree t(k, names);
    bl_out.assign((size_t)t.n_nodes * 3, 0.1);
    for (const auto& e : edges) {
        int a = tok2new.at(std::get<0>(e)), b = tok2new.at(std::get<1>(e));
        t.attach(a, b);
        Tree::bl_set(t.neighbors, bl_out.data(), a, b, std::get<2>(e));
    }
    return t;
}
