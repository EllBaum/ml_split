// test_merge_prep.cpp — detach a clade, regraft it, recover the original
// tree's splits AND branch lengths exactly.
#include "merge_prep.h"
#include <iostream>
#include <functional>
#include <cmath>

static int failures = 0;

// Canonical signature: split (smaller side's sorted leaf names) -> branch length.
static std::map<std::string,double> signature(const Tree& t, const double* bl) {
    std::map<std::string,double> sig;
    std::vector<std::string> all = t.taxon_names;
    std::sort(all.begin(), all.end());
    std::function<std::set<std::string>(int,int)> dfs =
        [&](int node, int parent) -> std::set<std::string> {
            std::set<std::string> below;
            if (t.is_leaf(node)) below.insert(t.taxon_names[node]);
            int nb[3]; int nc = t.neighbors_of(node, nb);
            for (int i = 0; i < nc; ++i) if (nb[i] != parent) {
                auto s = dfs(nb[i], node); below.insert(s.begin(), s.end());
            }
            if (parent != INVALID) {
                double w = Tree::bl_get(t.neighbors, bl, node, parent);
                std::set<std::string> side = below, comp;
                for (auto& nm : all) if (!side.count(nm)) comp.insert(nm);
                if (side.size() > comp.size() ||
                    (side.size() == comp.size() && comp < side)) side = comp;
                std::string key;
                for (auto& nm : side) key += nm + "|";
                sig[key] = w;
            }
            return below;
        };
    dfs(t.n_taxa, INVALID);
    return sig;
}

static void roundtrip(const std::string& label, const std::string& nwk,
                      const std::set<std::string>& clade) {
    std::cerr << "[" << label << "] clade={";
    for (auto& s : clade) std::cerr << s << " ";
    std::cerr << "}\n";
    std::vector<double> tmp(4096, 0.1);
    Tree t0 = Tree::from_newick(nwk, tmp.data());
    std::vector<double> bl0(t0.n_nodes * 3, 0.1);
    Tree torig = Tree::from_newick(nwk, bl0.data(), t0.taxon_names);
    auto sig0 = signature(torig, bl0.data());

    AdjTree a = AdjTree::from(torig, bl0.data());
    DetachedClade d = detach_clade(a, clade);

    std::vector<double> blA;
    Tree A = a.to_tree(blA);
    std::cerr << "      A' = " << A.to_newick(blA.data()) << "\n";

    regraft_clade(a, d);
    std::vector<double> blR;
    Tree R = a.to_tree(blR);
    auto sigR = signature(R, blR.data());

    bool same_keys = (sig0.size() == sigR.size());
    bool same_bls  = true;
    for (auto& kv : sig0) {
        auto it = sigR.find(kv.first);
        if (it == sigR.end()) same_keys = false;
        else if (std::abs(it->second - kv.second) > 1e-12) same_bls = false;
    }
    bool ok = same_keys && same_bls && (R.n_taxa == torig.n_taxa);
    std::cerr << (ok ? "  ok   " : "  FAIL ")
              << "round-trip recovers splits+BLs"
              << (same_keys ? "" : " [SPLIT MISMATCH]")
              << (same_bls  ? "" : " [BL MISMATCH]") << "\n";
    if (!ok) {
        ++failures;
        std::cerr << "      orig: " << torig.to_newick(bl0.data()) << "\n";
        std::cerr << "      rgft: " << R.to_newick(blR.data()) << "\n";
    }
}

int main() {
    roundtrip("singleton", "((X:0.9,A:0.15):0.25,B:0.3,C:0.4);", {"X"});
    roundtrip("cherry",    "((O1:0.3,O2:0.4):0.6,A:0.2,(B:0.1,Cc:0.1):0.5);", {"O1","O2"});
    roundtrip("triple",
        "(((I1:0.2,(I2:0.3,I3:0.35):0.15):0.4,R1:0.5):0.3,R2:0.6,(R3:0.1,R4:0.1):0.2);",
        {"I1","I2","I3"});
    std::cerr << "\n" << (failures ? "FAILURES: " : "ALL PASS (") << failures
              << (failures ? "" : ")") << "\n";
    return failures ? 1 : 0;
}
