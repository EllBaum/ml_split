// test_split_choice.cpp — choose_split_subedge must return the sub-edge that
// gives the higher likelihood when the WHOLE clade is attached there, and it must
// attach the entire clade (not one leaf).
#include "merge_session.h"
#include "merge_prep.h"
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

// local copy of the token-offset (offset_clade is file-local in merge_session.cpp)
static DetachedClade offset_local(const DetachedClade& d, int off) {
    DetachedClade o;
    o.connect_token = d.connect_token + off;
    o.stem_bl = d.stem_bl; o.half_a = d.half_a; o.half_b = d.half_b;
    for (auto& e : d.edges)
        o.edges.push_back({std::get<0>(e)+off, std::get<1>(e)+off, std::get<2>(e)});
    for (auto& kv : d.token_name) o.token_name[kv.first+off] = kv.second;
    return o;
}

// replicate the scoring of attaching the clade on (a,b) — for an independent check
static double score_on(const AdjTree& base, const DetachedClade& clade,
                       int a, int b, const MSA& full, SubstModel& model,
                       std::set<std::string>* leaves_out = nullptr) {
    AdjTree cand = base;
    regraft_clade_on(cand, clade, a, b, 0.5);
    std::vector<double> cbl;
    Tree ct = cand.to_tree(cbl);
    if (leaves_out) leaves_out->insert(ct.taxon_names.begin(), ct.taxon_names.end());
    MSA cmsa = slice_msa(full, ct.taxon_names);
    LikelihoodScorer cs(ct, cmsa, model, cbl.data());
    return cs.score();
}

int main() {
    MSA full = MSA::from_fasta("tests/test.fas", false);   // A0..A3, B0..B3
    auto model = std::make_unique<JTT>();

    // search tree over 6 taxa; B2,B3 are spare (used for the clade)
    std::vector<double> t1(1<<16,0.1);
    Tree s0 = Tree::from_newick("(((A0:0.2,A1:0.2):0.1,(A2:0.2,A3:0.2):0.1):0.1,B0:0.3,B1:0.3);", t1.data());
    std::vector<double> sbl(s0.n_nodes*3,0.1);
    Tree S = Tree::from_newick("(((A0:0.2,A1:0.2):0.1,(A2:0.2,A3:0.2):0.1):0.1,B0:0.3,B1:0.3);", sbl.data(), s0.taxon_names);
    AdjTree Ms = AdjTree::from(S, sbl.data());

    // pick an internal node `mid` and two of its neighbors X,Y as the sub-edges
    int mid=-1,X=-1,Y=-1;
    for (int n=S.n_taxa;n<S.n_nodes && mid<0;++n){ int nb[3]; int nc=S.neighbors_of(n,nb);
        if(nc==3){ mid=n; X=nb[0]; Y=nb[1]; } }
    std::cerr << "sub-edges share mid=" << mid << ": (" << X << "," << mid << ") and ("
              << mid << "," << Y << ")\n";

    // build a (B2,B3) cherry clade via detach, then offset tokens out of Ms's range
    std::vector<double> h1(4096,0.1);
    Tree h0 = Tree::from_newick("((B2:0.3,B3:0.3):0.2,ZA:0.1,ZB:0.1);", h1.data());
    std::vector<double> hbl(h0.n_nodes*3,0.1);
    Tree H = Tree::from_newick("((B2:0.3,B3:0.3):0.2,ZA:0.1,ZB:0.1);", hbl.data(), h0.taxon_names);
    AdjTree Ha = AdjTree::from(H, hbl.data());
    DetachedClade cl = detach_clade(Ha, {"B2","B3"});
    DetachedClade clo = offset_local(cl, 1000);

    // function under test
    auto chosen = choose_split_subedge(Ms, clo, X, mid, Y, full, *model);

    // independent scores + whole-clade check
    std::set<std::string> leaves1;
    double ll1 = score_on(Ms, clo, X, mid, full, *model, &leaves1);
    double ll2 = score_on(Ms, clo, mid, Y, full, *model);
    std::cerr << "ll(X,mid)=" << ll1 << "  ll(mid,Y)=" << ll2 << "\n";

    auto argmax = (ll1 >= ll2) ? std::make_pair(X,mid) : std::make_pair(mid,Y);
    check(chosen == argmax, "chose the higher-likelihood sub-edge");
    check(leaves1.count("B2") && leaves1.count("B3"),
          "whole clade attached (both B2 and B3 present)");
    check(std::abs(ll1-ll2) > 1e-9, "the two sub-edges give different LL (real choice)");

    std::cerr << "\n" << (failures?"FAILURES: ":"ALL PASS (") << failures
              << (failures?"":")") << "\n";
    return failures?1:0;
}
