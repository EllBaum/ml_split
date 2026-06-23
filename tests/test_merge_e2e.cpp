// test_merge_e2e.cpp — run_merge end to end: prune interest outgroups, sweep the
// 14x14 window, assemble. The returned tree's independent full-tree score() must
// equal the reported joined LL (no inherited outgroups here, so M = realA∪realB).
#include "merge_session.h"
#include "likelihood_scorer.h"
#include "jtt.h"
#include <iostream>
#include <memory>
#include <set>
#include <cmath>

static int failures = 0;
static void check(bool ok, const std::string& m) {
    std::cerr << (ok ? "  ok   " : "  FAIL ") << m << "\n";
    if (!ok) ++failures;
}

int main() {
    MSA full = MSA::from_fasta("tests/test.fas", false);   // A0..A3, B0..B3
    auto model = std::make_unique<JTT>();

    MergeInput in;
    in.newick_a = "((A0:0.2,A1:0.2):0.1,(A2:0.2,A3:0.2):0.1,O_A:0.5);";
    in.newick_b = "((B0:0.2,B1:0.2):0.1,(B2:0.2,B3:0.2):0.1,O_B:0.5);";
    in.interest_a = "O_A";
    in.interest_b = "O_B";
    // no inherited outgroups

    MergeResult res = run_merge(in, full, *model);
    std::cerr << "merged newick: " << res.newick << "\n";
    std::cerr << "reported LL  : " << res.loglik << "\n";

    // Parse the merged tree and score it independently.
    std::vector<double> tmp(1 << 16, 0.1);
    Tree m0 = Tree::from_newick(res.newick, tmp.data());
    std::vector<double> mbl(m0.n_nodes * 3, 0.1);
    Tree M = Tree::from_newick(res.newick, mbl.data(), m0.taxon_names);

    std::set<std::string> leaves(M.taxon_names.begin(), M.taxon_names.end());
    std::set<std::string> expect = {"A0","A1","A2","A3","B0","B1","B2","B3"};
    check(leaves == expect, "merged tree has exactly realA ∪ realB (8 taxa)");
    check(M.validate(), "merged tree validates");

    MSA msaM = slice_msa(full, M.taxon_names);
    LikelihoodScorer S(M, msaM, *model, mbl.data());
    double ll_indep = S.score();
    std::cerr << "independent score() of merged tree: " << ll_indep << "\n";

    double diff = std::abs(ll_indep - res.loglik);
    check(std::isfinite(res.loglik) && diff < 1e-6,
          "reported LL matches independent score (|diff|=" + std::to_string(diff) + ")");

    std::cerr << "\n" << (failures ? "FAILURES: " : "ALL PASS (") << failures
              << (failures ? "" : ")") << "\n";
    return failures ? 1 : 0;
}
