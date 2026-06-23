// test_pruned.cpp — exercise Tree::pruned against the worked examples.
#include "tree.h"
#include <iostream>
#include <set>
#include <cmath>

static int failures = 0;
static void check(bool ok,const std::string& msg) {
    std::cerr << (ok ? "  ok   " : "  FAIL ") << msg << "\n";
    if (!ok) ++failures;
}
static std::pair<int,int> norm(std::pair<int,int> e) {
    if (e.first > e.second) std::swap(e.first,e.second);
    return e;
}

// Build a Tree + bl from a Newick literal.
struct Built { Tree t; std::vector<double> bl; };
static Built build(const std::string& nwk) {
    // n_nodes unknown until parsed; from_newick needs bl_out pre-sized.
    // Count taxa cheaply by re-using from_newick twice is wasteful; instead
    // over-allocate using a quick taxa count.
    int commas_plus = 1; // leaves = (#leaf-name tokens); easier: parse then size.
    (void)commas_plus;
    // Two-pass: first parse into a temporary with a generous buffer.
    std::vector<double> tmp(4096,0.1);
    Tree t = Tree::from_newick(nwk,tmp.data());
    std::vector<double> bl(t.n_nodes * 3,0.1);
    // Re-parse into correctly sized bl (taxa_order fixed to keep ids stable).
    Tree t2 = Tree::from_newick(nwk,bl.data(),t.taxon_names);
    return { std::move(t2),std::move(bl) };
}


int main() {
    // ── Test A: branch-length summation + basic prune ─────────────────────────
    std::cerr << "[A] BL summation 0.15+0.25 -> 0.40\n";
    {
        auto b = build("((X:0.9,A:0.15):0.25,B:0.3,C:0.4);");
        auto r = b.t.pruned({"X"},b.bl.data());
        check(r.tree.n_taxa == 3,"3 leaves remain");
        // A is now joined to the surviving internal by 0.15+0.25 = 0.40
        // find A's only neighbor (internal) and its bl
        int ia = -1; for (int i=0;i<r.tree.n_taxa;++i) if (r.tree.taxon_names[i]=="A") ia=i;
        int nb[3]; r.tree.neighbors_of(ia,nb);
        double w = Tree::bl_get(r.tree.neighbors,r.bl.data(),ia,nb[0]);
        check(std::abs(w - 0.40) < 1e-12,"A-internal bl == 0.40 (got " + std::to_string(w) + ")");
        // anchor for X should be the (A,internal) edge
        auto e = norm(r.anchors[0]);
        std::pair<int,int> aedge = norm({ia,nb[0]});
        check(e == aedge,"anchor(X) is the A-internal edge");
        check(r.tree.validate(),"result validates");
        std::cerr << "      newick: " << r.tree.to_newick(r.bl.data()) << "\n";
    }

    // ── Test B: outgroups strung along one branch -> one shared anchor ────────
    std::cerr << "[B] strung-along O1,O2 collapse to one anchor edge\n";
    {
        auto b = build("((L1:0.1,L2:0.1):0.3,O1:0.5,(O2:0.5,(R1:0.1,R2:0.1):0.3):0.4);");
        auto r = b.t.pruned({"O1","O2"},b.bl.data());
        check(r.tree.n_taxa == 4,"4 leaves remain (L1,L2,R1,R2)");
        auto e0 = norm(r.anchors[0]),e1 = norm(r.anchors[1]);
        check(e0 == e1,"O1 and O2 share one anchor edge");
        // shared edge connects the two internal nodes; bl = 0.3+0.4+0.3 = 1.0
        bool both_internal = e0.first >= r.tree.n_taxa && e0.second >= r.tree.n_taxa;
        check(both_internal,"anchor edge is the internal a-b branch");
        double w = Tree::bl_get(r.tree.neighbors,r.bl.data(),e0.first,e0.second);
        check(std::abs(w - 1.0) < 1e-12,"merged branch bl == 1.0 (got " + std::to_string(w) + ")");
        std::cerr << "      newick: " << r.tree.to_newick(r.bl.data()) << "\n";
    }

    // ── Test C: outgroups in different regions -> distinct anchors ────────────
    std::cerr << "[C] distinct-region O1,O2 -> distinct anchors\n";
    {
        auto b = build("(((L1:0.1,L2:0.1):0.2,O1:0.5):0.3,C1:0.4,"
                       "(O2:0.5,(R1:0.1,R2:0.1):0.2):0.3);");
        auto r = b.t.pruned({"O1","O2"},b.bl.data());
        check(r.tree.n_taxa == 5,"5 leaves remain");
        auto e0 = norm(r.anchors[0]),e1 = norm(r.anchors[1]);
        check(e0 != e1,"O1 and O2 have different anchor edges");
        std::cerr << "      newick: " << r.tree.to_newick(r.bl.data()) << "\n";
    }

    // ── Test D: cherry of outgroups cascades to a shared anchor ───────────────
    std::cerr << "[D] cherry (O1,O2) cascade -> shared anchor\n";
    {
        auto b = build("((O1:0.3,O2:0.4):0.6,A:0.2,(B:0.1,Cc:0.1):0.5);");
        auto r = b.t.pruned({"O1","O2"},b.bl.data());
        check(r.tree.n_taxa == 3,"3 leaves remain (A,B,Cc)");
        auto e0 = norm(r.anchors[0]),e1 = norm(r.anchors[1]);
        check(e0 == e1,"cherry outgroups share one anchor");
        // anchor should be the A-internal edge; bl = 0.2 + 0.5 = 0.7
        double w = Tree::bl_get(r.tree.neighbors,r.bl.data(),e0.first,e0.second);
        check(std::abs(w - 0.7) < 1e-12,"collapsed branch bl == 0.7 (got " + std::to_string(w) + ")");
        std::cerr << "      newick: " << r.tree.to_newick(r.bl.data()) << "\n";
    }

    std::cerr << "\n" << (failures ? "FAILURES: " : "ALL PASS (")
              << failures << (failures ? "" : ")") << "\n";
    return failures ? 1 : 0;
}
