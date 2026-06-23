// test_scorer_smoke.cpp — confirm LikelihoodScorer compiles + runs in this repo.
#include "tree.h"
#include "msa.h"
#include "subst_model.h"
#include "jtt.h"
#include "likelihood_scorer.h"
#include <iostream>
#include <memory>

int main() {
    MSA msa = MSA::from_fasta("tests/test.fas", false);
    std::cerr << "MSA: n_taxa=" << msa.n_taxa
              << " n_patterns=" << msa.n_patterns << "\n";

    // caterpillar newick over the 8 taxa, no spaces
    std::string nwk =
        "((((((A0:0.1,A1:0.1):0.1,A2:0.1):0.1,A3:0.1):0.1,B0:0.1):0.1,B1:0.1):0.1,B2:0.1,B3:0.1);";
    std::vector<double> bl(2 * msa.n_taxa * 3, 0.1);  // generous; from_newick sizes to n_nodes*3
    std::vector<double> tmp(4096, 0.1);
    Tree t = Tree::from_newick(nwk, tmp.data());
    std::vector<double> bl2(t.n_nodes * 3, 0.1);
    Tree tree = Tree::from_newick(nwk, bl2.data(), t.taxon_names);

    std::unique_ptr<SubstModel> model = std::make_unique<JTT>();
    LikelihoodScorer scorer(tree, msa, *model, bl2.data());
    std::cerr << "initial score = " << scorer.score() << "\n";

    double opt = scorer.optimize_branch_lengths("fast", 10.0, 32);
    std::cerr << "after fast BLO = " << opt << "\n";
    std::cerr << (std::isfinite(opt) ? "ok   scorer builds and runs\n"
                                     : "FAIL non-finite score\n");
    return std::isfinite(opt) ? 0 : 1;
}
