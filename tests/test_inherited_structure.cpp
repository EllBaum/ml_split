// test_inherited_structure.cpp — regression for inherited-clade STRUCTURE
// preservation. The interest outgroup O_A is nested *between* two inherited
// leaves, so the inherited set's spanning subtree in the raw newick includes
// O_A. The merge must remove the interest first, recognize (IA1,IA2) as one
// connected inherited clade, and restore it with its cherry intact — so the
// bipartition {IA1,IA2} survives in the merged tree. (LL-only tests missed
// this; a scrambled inherited clade breaks the next round's pointer.)
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

// Collect each internal edge's descendant-leaf set (rooted parse) as a check
// for whether {IA1,IA2} forms a clade in the merged tree.
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
            out.push_back(leaves);                 // internal node = one clade
        } else {
            size_t st = pos;
            while (pos < s.size() && s[pos] != ',' && s[pos] != '(' &&
                   s[pos] != ')' && s[pos] != ':') ++pos;
            leaves.insert(s.substr(st, pos - st));
        }
        if (pos < s.size() && s[pos] == ':') {     // skip branch length
            ++pos;
            while (pos < s.size() && s[pos] != ',' && s[pos] != ')') ++pos;
        }
        return leaves;
    };
    rec();
}

int main() {
    // Build an alignment that includes the inherited taxa (the split-edge case
    // scores whole clades, so their sequences must be present — as they always
    // are in real use, where the full MSA carries every taxon).
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
    // add IA1, IA2 rows (reuse existing sequences; identity is irrelevant here)
    auto seq_of = [&](const std::string& n){ for (size_t i=0;i<names.size();++i) if (names[i]==n) return seqs[i]; return std::string(); };
    names.push_back("IA1"); seqs.push_back(seq_of("A2"));
    names.push_back("IA2"); seqs.push_back(seq_of("A3"));
    MSA full = MSA::from_sequences(names, seqs, false);
    auto model = std::make_unique<JTT>();

    MergeInput in;
    // A: realA + interest O_A nested BETWEEN inherited IA1 and IA2.
    // Spanning subtree of {IA1,IA2} = (IA1,(O_A,IA2)) — it contains O_A.
    in.newick_a = "((A0:0.2,A1:0.2):0.1,(A2:0.2,A3:0.2):0.1,"
                  "(IA1:0.3,(O_A:0.3,IA2:0.3):0.2):0.2);";
    in.newick_b = "((B0:0.2,B1:0.2):0.1,(B2:0.2,B3:0.2):0.1,O_B:0.5);";
    in.interest_a = "O_A";
    in.interest_b = "O_B";
    in.inherited_a = {{"IA1", "IA2"}};   // passed together…
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
    std::set<std::string> want = {"IA1", "IA2"};
    bool cherry = false;
    for (auto& c : cl) if (c == want) { cherry = true; break; }
    check(cherry, "inherited clade {IA1,IA2} preserved as a bipartition");

    // Same scenario, inherited passed as separate singletons — auto-detect must
    // regroup them into the one connected clade and still preserve the cherry.
    MergeInput in2 = in;
    in2.inherited_a = {{"IA1"}, {"IA2"}};
    MergeResult res2 = run_merge(in2, full, *model);
    std::vector<std::set<std::string>> cl2;
    clades(res2.newick, cl2);
    bool cherry2 = false;
    for (auto& c : cl2) if (c == want) { cherry2 = true; break; }
    check(cherry2, "cherry preserved even when inherited passed as singletons");

    std::cerr << "\n" << (failures ? "FAILURES: " : "ALL PASS (") << failures
              << (failures ? "" : ")") << "\n";
    return failures ? 1 : 0;
}
