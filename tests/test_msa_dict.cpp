// test_msa_dict.cpp — from_sequences (dict-style input) must produce the same
// patterns/weights as from_fasta on identical data.
#include "msa.h"
#include <iostream>
#include <fstream>
#include <sstream>

static int failures = 0;
static void check(bool ok, const std::string& m) {
    std::cerr << (ok ? "  ok   " : "  FAIL ") << m << "\n";
    if (!ok) ++failures;
}

int main() {
    // minimal FASTA reader for the test fixture
    std::vector<std::string> names, seqs;
    {
        std::ifstream f("tests/test.fas");
        std::string line, cur;
        while (std::getline(f, line)) {
            if (!line.empty() && line[0] == '>') {
                if (!cur.empty()) { seqs.push_back(cur); cur.clear(); }
                names.push_back(line.substr(1));
            } else cur += line;
        }
        if (!cur.empty()) seqs.push_back(cur);
    }
    std::cerr << "read " << names.size() << " seqs from test.fas\n";

    MSA viaFile = MSA::from_fasta("tests/test.fas", false);
    MSA viaSeq  = MSA::from_sequences(names, seqs, false);

    check(viaSeq.n_taxa == viaFile.n_taxa, "n_taxa matches");
    check(viaSeq.n_patterns == viaFile.n_patterns, "n_patterns matches");
    check(viaSeq.weights == viaFile.weights, "weights match");
    check(viaSeq.taxon_names == viaFile.taxon_names, "taxon order matches");
    bool data_ok = (viaSeq.data.size() == viaFile.data.size());
    if (data_ok)
        for (size_t i = 0; i < viaSeq.data.size(); ++i)
            if (viaSeq.data[i] != viaFile.data[i]) { data_ok = false; break; }
    check(data_ok, "encoded pattern data matches");

    std::cerr << "\n" << (failures ? "FAILURES: " : "ALL PASS (") << failures
              << (failures ? "" : ")") << "\n";
    return failures ? 1 : 0;
}
