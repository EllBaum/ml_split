// test_merge_util.cpp — slice_msa column-alignment + window_edges BFS.
#include "merge_session.h"
#include <iostream>
#include <set>

static int failures = 0;
static void check(bool ok, const std::string& m) {
    std::cerr << (ok ? "  ok   " : "  FAIL ") << m << "\n";
    if (!ok) ++failures;
}

int main() {
    // ── slice_msa ─────────────────────────────────────────────────────────────
    std::cerr << "[slice_msa]\n";
    MSA full = MSA::from_fasta("tests/test.fas", false);
    std::vector<std::string> aN = {"A0","A1","A2","A3"};
    std::vector<std::string> bN = {"B0","B1","B2","B3"};
    MSA A = slice_msa(full, aN);
    MSA B = slice_msa(full, bN);

    check(A.n_patterns == full.n_patterns && B.n_patterns == full.n_patterns,
          "slices keep full pattern count (column-aligned)");
    check(A.weights == full.weights && B.weights == full.weights,
          "slices keep full weights");
    check(A.n_taxa == 4 && B.n_taxa == 4, "row counts correct");
    check(A.taxon_names == aN && B.taxon_names == bN, "row order preserved");

    // data rows match the originals
    std::unordered_map<std::string,int> row;
    for (int i = 0; i < full.n_taxa; ++i) row[full.taxon_names[i]] = i;
    bool data_ok = true;
    for (int i = 0; i < A.n_taxa; ++i)
        for (int p = 0; p < A.n_patterns; ++p)
            if (A.at(i,p) != full.at(row[aN[i]], p)) data_ok = false;
    check(data_ok, "sliced data matches source rows");

    // ── window_edges ────────────────────────────────────────────────────────────
    std::cerr << "[window_edges]\n";
    std::string nwk =
        "((((((A0:0.1,A1:0.1):0.1,A2:0.1):0.1,A3:0.1):0.1,B0:0.1):0.1,B1:0.1):0.1,B2:0.1,B3:0.1);";
    std::vector<double> tmp(4096,0.1);
    Tree t0 = Tree::from_newick(nwk, tmp.data());
    std::vector<double> bl(t0.n_nodes*3,0.1);
    Tree t = Tree::from_newick(nwk, bl.data(), t0.taxon_names);
    int total_edges = t.n_edges;          // 2*8-3 = 13

    // pick an internal edge as anchor
    int a0=-1,a1=-1;
    for (int n=t.n_taxa;n<t.n_nodes && a0<0;++n){ int nb[3]; t.neighbors_of(n,nb);
        for(int i=0;i<3;++i) if(nb[i]>=t.n_taxa){a0=n;a1=nb[i];break;} }

    auto w14 = window_edges(t, a0, a1, 14);
    std::set<std::pair<int,int>> s14(w14.begin(), w14.end());
    auto norm=[](int u,int v){return u<v?std::make_pair(u,v):std::make_pair(v,u);};
    check((int)w14.size() == total_edges, "window(14) returns all 13 edges of this tree");
    check(s14.size() == w14.size(), "no duplicate edges");
    check(s14.count(norm(a0,a1)) == 1, "window includes the anchor edge");
    bool all_real = true;
    for (auto& e : w14) if (!t.are_connected(e.first, e.second)) all_real = false;
    check(all_real, "all returned edges exist in the tree");

    auto w5 = window_edges(t, a0, a1, 5);
    std::set<std::pair<int,int>> s5(w5.begin(), w5.end());
    check((int)w5.size() == 5, "window(5) caps at 5");
    check(s5.count(norm(a0,a1)) == 1, "capped window still includes anchor");
    bool subset = true; for (auto& e : w5) if (!s14.count(e)) subset = false;
    check(subset, "capped window is a subset of the full window");

    std::cerr << "\n" << (failures?"FAILURES: ":"ALL PASS (") << failures
              << (failures?"":")") << "\n";
    return failures?1:0;
}
