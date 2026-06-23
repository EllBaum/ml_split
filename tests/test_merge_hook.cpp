// test_merge_hook.cpp — validate optimize_five_for_merge: the joined LL it
// reports must equal an independent full-tree score() at the same 5 branch
// lengths. Exercises the connector CLV-combination on a single known tree.
#include "tree.h"
#include "msa.h"
#include "subst_model.h"
#include "jtt.h"
#include "likelihood_scorer.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <memory>

int main() {
    MSA msa = MSA::from_fasta("tests/test.fas", false);
    std::string nwk =
        "((((((A0:0.3,A1:0.2):0.25,A2:0.15):0.2,A3:0.3):0.18,B0:0.22):0.21,B1:0.27):0.19,B2:0.24,B3:0.26);";
    std::vector<double> tmp(4096, 0.1);
    Tree t0 = Tree::from_newick(nwk, tmp.data());
    std::vector<double> bl(t0.n_nodes * 3, 0.1);
    Tree tree = Tree::from_newick(nwk, bl.data(), t0.taxon_names);

    auto model = std::make_unique<JTT>();
    LikelihoodScorer S(tree, msa, *model, bl.data());

    // Find an internal-internal edge (u,v); collect u's other two neighbors a,b
    // and v's other two c,d.
    int u = -1, v = -1, a = -1, b = -1, c = -1, d = -1;
    for (int n = tree.n_taxa; n < tree.n_nodes && u < 0; ++n) {
        int nb[3]; tree.neighbors_of(n, nb);
        for (int i = 0; i < 3; ++i) {
            int m = nb[i];
            if (m >= tree.n_taxa) {            // internal neighbor
                u = n; v = m;
                int ou[2]; tree.neighbors_except(u, v, ou); a = ou[0]; b = ou[1];
                int ov[2]; tree.neighbors_except(v, u, ov); c = ov[0]; d = ov[1];
                break;
            }
        }
    }
    std::cerr << "internal edge (u,v)=(" << u << "," << v << ")  u~{" << a << "," << b
              << "}  v~{" << c << "," << d << "}\n";

    // Boundary directional CLVs (frozen-subtree boundary conditions).
    std::vector<double> a_clv, a_ls, b_clv, b_ls, c_clv, c_ls, d_clv, d_ls;
    S.clv(a, u, a_clv, a_ls);
    S.clv(b, u, b_clv, b_ls);
    S.clv(c, v, c_clv, c_ls);
    S.clv(d, v, d_clv, d_ls);

    double init_bl5[5] = {
        S.branch_length(u, v),
        S.branch_length(u, a),
        S.branch_length(u, b),
        S.branch_length(v, c),
        S.branch_length(v, d),
    };
    double bl_out[5];
    double ll_hook = S.optimize_five_for_merge(
        a_clv.data(), a_ls.data(),
        b_clv.data(), b_ls.data(),
        c_clv.data(), c_ls.data(),
        d_clv.data(), d_ls.data(),
        init_bl5, bl_out, /*eps=*/1e-9);

    std::cerr << "hook LL = " << ll_hook << "\n  bl_out = {";
    for (int i = 0; i < 5; ++i) std::cerr << bl_out[i] << (i<4?", ":"}\n");

    // Write the 5 optimized BLs back into a fresh bl array, score independently.
    std::vector<double> bl2 = S.get_bl_array();   // flat n_nodes*3
    Tree::bl_set(tree.neighbors, bl2.data(), u, v, bl_out[0]);
    Tree::bl_set(tree.neighbors, bl2.data(), u, a, bl_out[1]);
    Tree::bl_set(tree.neighbors, bl2.data(), u, b, bl_out[2]);
    Tree::bl_set(tree.neighbors, bl2.data(), v, c, bl_out[3]);
    Tree::bl_set(tree.neighbors, bl2.data(), v, d, bl_out[4]);

    LikelihoodScorer S2(tree, msa, *model, bl2.data());
    double ll_ref = S2.score();
    std::cerr << "independent score() at those BLs = " << ll_ref << "\n";

    double diff = std::abs(ll_hook - ll_ref);
    bool ok = std::isfinite(ll_hook) && diff < 1e-6;
    std::cerr << (ok ? "  ok   " : "  FAIL ")
              << "hook LL matches independent score (|diff|=" << diff << ")\n";
    return ok ? 0 : 1;
}
