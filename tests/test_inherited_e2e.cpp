// test_inherited_e2e.cpp — run_merge with inherited outgroups. Verifies they are
// restored into the merged tree, and that pruning them back out recovers a tree
// whose score() equals the reported search LL (restore didn't disturb the search
// topology/BLs, and the BL-summation on prune undoes the restore's edge split).
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
    // A: realA + interest O_A + inherited singleton IA (anchored on a pendant edge)
    in.newick_a = "(((A0:0.2,IA:0.3):0.15,A1:0.2):0.1,(A2:0.2,A3:0.2):0.1,O_A:0.5);";
    // B: realB + interest O_B + inherited cherry (IB1,IB2)
    in.newick_b = "(((B0:0.2,(IB1:0.3,IB2:0.3):0.2):0.15,B1:0.2):0.1,(B2:0.2,B3:0.2):0.1,O_B:0.5);";
    in.interest_a = "O_A";
    in.interest_b = "O_B";
    in.inherited_a = {{"IA"}};
    in.inherited_b = {{"IB1","IB2"}};

    MergeResult res = run_merge(in, full, *model);
    std::cerr << "merged newick: " << res.newick << "\n";
    std::cerr << "reported LL  : " << res.loglik << "\n";

    std::vector<double> tmp(1 << 16, 0.1);
    Tree m0 = Tree::from_newick(res.newick, tmp.data());
    std::vector<double> mbl(m0.n_nodes * 3, 0.1);
    Tree M = Tree::from_newick(res.newick, mbl.data(), m0.taxon_names);

    std::set<std::string> leaves(M.taxon_names.begin(), M.taxon_names.end());
    std::set<std::string> expect = {"A0","A1","A2","A3","B0","B1","B2","B3",
                                    "IA","IB1","IB2"};
    check(leaves == expect, "merged tree has realA ∪ realB ∪ inherited (11 taxa)");
    check(M.validate(), "merged tree validates");

    // Prune the inherited back out; the search tree should re-emerge, and its
    // score must equal the reported (search-only) LL.
    PruneResult ps = M.pruned({"IA","IB1","IB2"}, mbl.data());
    std::set<std::string> sl(ps.tree.taxon_names.begin(), ps.tree.taxon_names.end());
    std::set<std::string> rs = {"A0","A1","A2","A3","B0","B1","B2","B3"};
    check(sl == rs, "removing inherited recovers realA ∪ realB");

    MSA msaS = slice_msa(full, ps.tree.taxon_names);
    LikelihoodScorer S(ps.tree, msaS, *model, ps.bl.data());
    double ll_search = S.score();
    std::cerr << "score(search tree recovered from M) = " << ll_search << "\n";

    double diff = std::abs(ll_search - res.loglik);
    check(std::isfinite(res.loglik) && diff < 1e-6,
          "reported LL matches recovered search tree score (|diff|=" +
          std::to_string(diff) + ")");

    std::cerr << "\n" << (failures ? "FAILURES: " : "ALL PASS (") << failures
              << (failures ? "" : ")") << "\n";
    return failures ? 1 : 0;
}
