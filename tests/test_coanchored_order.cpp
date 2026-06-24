// test_coanchored_order.cpp — regression for CO-ANCHORED inherited ordering.
//
// Two inherited singletons stacked along one path with reals only at the ends
// (A0—c1·IA1—c2·IA2—rest) both collapse onto the SAME anchor edge when pruned
// for the search (c1,c2 suppressed). The restore must put them back in their
// along-edge order — IA1 nearer A0, IA2 nearer the rest — otherwise the two
// swap and the side is no longer faithful (the next round's pointer breaks).
//
// Caught the OrthoMaM-200 corpus faithfulness failures (faith_a/faith_b == 2):
// the earlier code attached the second co-anchored clade on the wrong side.
#include "merge_session.h"
#include "likelihood_scorer.h"
#include "jtt.h"
#include <iostream>
#include <memory>
#include <set>
#include <vector>
#include <string>
#include <functional>
#include <fstream>

static int failures = 0;
static void check(bool ok, const std::string& m) {
    std::cerr << (ok ? "  ok   " : "  FAIL ") << m << "\n";
    if (!ok) ++failures;
}

// Descendant-leaf set of every internal node (rooted parse) — the bipartitions.
static void clades(const std::string& s, std::vector<std::set<std::string>>& out) {
    size_t pos = 0;
    std::function<std::set<std::string>()> rec = [&]() {
        std::set<std::string> leaves;
        if (s[pos] == '(') {
            ++pos;
            while (true) {
                auto sub = rec();
                leaves.insert(sub.begin(), sub.end());
                if (s[pos] == ',') { ++pos; continue; }
                if (s[pos] == ')') { ++pos; break; }
            }
            out.push_back(leaves);
        } else {
            size_t st = pos;
            while (pos < s.size() && s[pos] != ',' && s[pos] != '(' &&
                   s[pos] != ')' && s[pos] != ':') ++pos;
            leaves.insert(s.substr(st, pos - st));
        }
        if (pos < s.size() && s[pos] == ':') {
            ++pos;
            while (pos < s.size() && s[pos] != ',' && s[pos] != ')') ++pos;
        }
        return leaves;
    };
    rec();
}

static bool has_split(const std::vector<std::set<std::string>>& cl,
                      const std::set<std::string>& want,
                      const std::set<std::string>& all) {
    std::set<std::string> comp;
    for (auto& x : all) if (!want.count(x)) comp.insert(x);
    for (auto& c : cl) if (c == want || c == comp) return true;  // unrooted split
    return false;
}

int main() {
    std::vector<std::string> names, seqs;
    {
        std::ifstream f("tests/test.fas");
        std::string line, cur, nm;
        auto flush = [&]{ if (!nm.empty()) { names.push_back(nm); seqs.push_back(cur); } };
        while (std::getline(f, line)) {
            if (!line.empty() && line[0] == '>') { flush(); nm = line.substr(1); cur.clear(); }
            else cur += line;
        }
        flush();
    }
    auto seq_of = [&](const std::string& n){ for (size_t i=0;i<names.size();++i) if (names[i]==n) return seqs[i]; return std::string(); };
    names.push_back("IA1"); seqs.push_back(seq_of("A2"));
    names.push_back("IA2"); seqs.push_back(seq_of("A3"));
    MSA full = MSA::from_sequences(names, seqs, false);
    auto model = std::make_unique<JTT>();

    MergeInput in;
    // A: caterpillar  A0 — c1(·IA1) — c2(·IA2) — {A1,(A2,A3)} ,  O_A off the base.
    // Pruning IA1+IA2 for the search collapses c1,c2 → both anchor on (A0,rest).
    // IA1 must come back nearer A0, IA2 nearer the rest.
    in.newick_a = "((((A0:.2,IA1:.3):.1,IA2:.3):.1,A1:.2):.1,(A2:.2,A3:.2):.1,O_A:.5);";
    in.newick_b = "((B0:.2,B1:.2):.1,(B2:.2,B3:.2):.1,O_B:.5);";
    in.interest_a = "O_A";
    in.interest_b = "O_B";
    in.inherited_a = {{"IA1"}, {"IA2"}};   // singletons → auto-detect path
    in.inherited_b = {};

    MergeResult res = run_merge(in, full, *model);
    std::cerr << "merged: " << res.newick << "\n";

    std::vector<double> tmp(1 << 16, 0.1);
    Tree m0 = Tree::from_newick(res.newick, tmp.data());
    std::set<std::string> leaves(m0.taxon_names.begin(), m0.taxon_names.end());
    std::set<std::string> expect = {"A0","A1","A2","A3","B0","B1","B2","B3","IA1","IA2"};
    check(leaves == expect, "merged has realA ∪ realB ∪ {IA1,IA2}; O_A/O_B gone");

    std::vector<std::set<std::string>> cl;
    clades(res.newick, cl);
    // Order-preserving: IA1 sits with A0 (split {A0,IA1}); IA2 does NOT.
    check( has_split(cl, {"A0","IA1"}, expect), "IA1 restored nearer A0 (split {A0,IA1} present)");
    check(!has_split(cl, {"A0","IA2"}, expect), "IA2 NOT swapped to A0 (no {A0,IA2} split)");

    std::cerr << "\n" << (failures ? "FAILURES: " : "ALL PASS (") << failures
              << (failures ? "" : ")") << "\n";
    return failures ? 1 : 0;
}
