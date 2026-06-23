// msa.cpp — Implementation of MSA

#include "msa.h"
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <cctype>
#include <cstring>

// DNA encoding table

uint32_t dna_encode(char ch) {
    switch (std::toupper(ch)) {
        case 'A': return DNA_A;
        case 'C': return DNA_C;
        case 'G': return DNA_G;
        case 'T': case 'U': return DNA_T;
        case 'R': return DNA_A | DNA_G;
        case 'Y': return DNA_C | DNA_T;
        case 'M': return DNA_A | DNA_C;
        case 'K': return DNA_G | DNA_T;
        case 'S': return DNA_C | DNA_G;
        case 'W': return DNA_A | DNA_T;
        case 'B': return DNA_C | DNA_G | DNA_T;
        case 'D': return DNA_A | DNA_G | DNA_T;
        case 'H': return DNA_A | DNA_C | DNA_T;
        case 'V': return DNA_A | DNA_C | DNA_G;
        case 'N': return DNA_A | DNA_C | DNA_G | DNA_T;
        case '-': case '.': case '?': return DNA_ALL;
        default: return DNA_ALL;
    }
}

// AA encoding table

// Alphabetical order: ACDEFGHIKLMNPQRSTVWY (indices 0-19)
static const char AA_ORDER[] = "ACDEFGHIKLMNPQRSTVWY";

static int aa_char_to_index(char ch) {
    for (int i = 0; i < 20; ++i)
        if (AA_ORDER[i] == ch) return i;
    return -1;
}

uint32_t aa_encode(char ch) {
    ch = static_cast<char>(std::toupper(ch));
    int idx = aa_char_to_index(ch);
    if (idx >= 0) return 1u << idx;

    switch (ch) {
        case 'B': return (1u << aa_char_to_index('D')) | (1u << aa_char_to_index('N'));
        case 'Z': return (1u << aa_char_to_index('E')) | (1u << aa_char_to_index('Q'));
        case 'J': return (1u << aa_char_to_index('I')) | (1u << aa_char_to_index('L'));
        case 'X': return AA_ALL_STATES;
        case '-': case '.': case '?': case '*': return AA_ALL;
        default: return AA_ALL;
    }
}

// FASTA parsing

struct RawMSA {
    std::vector<std::string> names;
    std::vector<std::string> seqs;
};

static RawMSA read_fasta(const std::string& path) {
    RawMSA raw;
    std::ifstream in(path);
    if (!in.is_open())
        throw std::runtime_error("Cannot open FASTA file: " + path);

    std::string line;
    std::string current_seq;

    while (std::getline(in, line)) {
        // Strip trailing whitespace
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' '))
            line.pop_back();

        if (line.empty()) continue;

        if (line[0] == '>') {
            if (!current_seq.empty()) {
                raw.seqs.push_back(std::move(current_seq));
                current_seq.clear();
            }
            // Name: after '>' up to first whitespace
            std::string name = line.substr(1);
            // Trim leading space
            size_t start = name.find_first_not_of(' ');
            if (start != std::string::npos) name = name.substr(start);
            // Take first word
            size_t sp = name.find(' ');
            if (sp != std::string::npos) name = name.substr(0, sp);
            raw.names.push_back(name);
        } else {
            // Strip spaces within sequence
            for (char c : line) {
                if (c != ' ') current_seq += c;
            }
        }
    }
    if (!current_seq.empty())
        raw.seqs.push_back(std::move(current_seq));

    if (raw.names.size() != raw.seqs.size())
        throw std::runtime_error("Name/sequence count mismatch in FASTA");
    if (raw.names.size() < 3)
        throw std::runtime_error("Need >= 3 taxa, got " + std::to_string(raw.names.size()));

    size_t len = raw.seqs[0].size();
    if (len == 0)
        throw std::runtime_error("Empty sequences in FASTA");
    for (size_t i = 1; i < raw.seqs.size(); ++i) {
        if (raw.seqs[i].size() != len)
            throw std::runtime_error("Unequal sequence lengths: " +
                std::to_string(len) + " vs " + std::to_string(raw.seqs[i].size()));
    }

    return raw;
}


// Auto-detect sequence type

static SeqType detect_type(const std::vector<std::string>& seqs) {
    static const char aa_only[] = "EFIJLOPQZX*";

    for (const auto& seq : seqs) {
        for (char c : seq) {
            char upper = static_cast<char>(std::toupper(c));
            for (const char* p = aa_only; *p; ++p) {
                if (upper == *p) return SeqType::AA;
            }
        }
    }
    return SeqType::DNA;
}

// Encoding

// Encode into flat array [n_taxa * n_sites], taxa-major
static std::vector<uint32_t> encode(const std::vector<std::string>& seqs,
                                     SeqType st, int n_taxa, int n_sites) {
    std::vector<uint32_t> data(n_taxa * n_sites);
    for (int t = 0; t < n_taxa; ++t) {
        for (int s = 0; s < n_sites; ++s) {
            char ch = seqs[t][s];
            data[t * n_sites + s] = (st == SeqType::DNA) ? dna_encode(ch)
                                                          : aa_encode(ch);
        }
    }
    return data;
}

// Pattern deduplication

struct DeduplicateResult {
    std::vector<uint32_t> patterns;  // n_taxa * n_unique, taxa-major
    std::vector<int> weights;
    int n_unique;
};

static DeduplicateResult deduplicate(const std::vector<uint32_t>& encoded,
                                      int n_taxa, int n_sites) {
    std::unordered_map<std::string, int> col_map;
    std::vector<int> pattern_cols;
    std::vector<int> wts;

    std::string col_key(n_taxa * sizeof(uint32_t), '\0');

    for (int s = 0; s < n_sites; ++s) {
        // Build column key
        for (int t = 0; t < n_taxa; ++t) {
            uint32_t val = encoded[t * n_sites + s];
            std::memcpy(&col_key[t * sizeof(uint32_t)], &val, sizeof(uint32_t));
        }

        auto it = col_map.find(col_key);
        if (it != col_map.end()) {
            wts[it->second]++;
        } else {
            col_map[col_key] = static_cast<int>(pattern_cols.size());
            pattern_cols.push_back(s);
            wts.push_back(1);
        }
    }

    int n_unique = static_cast<int>(pattern_cols.size());
    std::vector<uint32_t> patterns(n_taxa * n_unique);
    for (int p = 0; p < n_unique; ++p) {
        int s = pattern_cols[p];
        for (int t = 0; t < n_taxa; ++t) {
            patterns[t * n_unique + p] = encoded[t * n_sites + s];
        }
    }

    return {std::move(patterns), std::move(wts), n_unique};
}

// Parsimony-informative check

static bool is_informative(const std::vector<uint32_t>& patterns,
                           int n_taxa, int n_unique, int col,
                           SeqType st) {
    uint32_t gap_bit = (st == SeqType::DNA) ? DNA_GAP : AA_GAP_BIT;
    uint32_t state_mask = gap_bit - 1;

    std::unordered_map<uint32_t, int> counts;
    int n_unambig = 0;

    for (int t = 0; t < n_taxa; ++t) {
        uint32_t val = patterns[t * n_unique + col];
        uint32_t bits = val & state_mask;
        // Unambiguous: exactly one state bit set (power of 2, nonzero)
        if (bits != 0 && (bits & (bits - 1)) == 0) {
            n_unambig++;
            counts[bits]++;
        }
    }

    // Conservatively keep all-ambiguous columns
    if (n_unambig == 0) return true;

    // Informative: >=2 states each appearing >=2 times
    int n_repeated = 0;
    for (const auto& [state, cnt] : counts) {
        if (cnt >= 2) n_repeated++;
    }
    return n_repeated >= 2;
}

// Remove uninformative patterns

struct FilterResult {
    std::vector<uint32_t> patterns;
    std::vector<int> weights;
    int n_patterns;
    int offset = 0;  // fixed parsimony cost of removed non-informative patterns
};

static FilterResult filter_to_parsimony_informative(const std::vector<uint32_t>& patterns,
                                          const std::vector<int>& weights,
                                          int n_taxa, int n_unique,
                                          SeqType st) {
    // gap sentinel: all-states bitmask (topology-independent, treated as missing)
    uint32_t gap_val = (st == SeqType::AA) ? 0x1FFFFFu : 0x1Fu;

    std::vector<int> keep;
    int offset = 0;

    for (int p = 0; p < n_unique; ++p) {
        if (is_informative(patterns, n_taxa, n_unique, p, st)) {
            keep.push_back(p);
        } else {
            // Compute topology-independent Fitch cost for this removed pattern.
            // Fold Fitch over non-gap taxa from left.
            uint32_t state = 0;
            bool started = false;
            int cost = 0;
            for (int t = 0; t < n_taxa; ++t) {
                uint32_t c = patterns[t * n_unique + p];
                if (c == gap_val) continue;
                if (!started) { state = c; started = true; continue; }
                uint32_t inter = state & c;
                if (inter == 0) { state = state | c; ++cost; }
                else             { state = inter; }
            }
            offset += weights[p] * cost;
        }
    }

    int n_keep = static_cast<int>(keep.size());
    std::vector<uint32_t> out_pat(n_taxa * n_keep);
    std::vector<int> out_wts(n_keep);

    for (int i = 0; i < n_keep; ++i) {
        int p = keep[i];
        out_wts[i] = weights[p];
        for (int t = 0; t < n_taxa; ++t) {
            out_pat[t * n_keep + i] = patterns[t * n_unique + p];
        }
    }

    return {std::move(out_pat), std::move(out_wts), n_keep, offset};
}

// Public API

// Shared builder: (names, seqs) → encoded → dedup → filter → MSA.
static MSA from_raw(const std::vector<std::string>& names,
                    const std::vector<std::string>& seqs,
                    bool parsimony_informative_only) {
    int n_taxa = static_cast<int>(names.size());
    if (n_taxa == 0 || seqs.empty())
        throw std::runtime_error("MSA: empty alignment");
    int n_sites = static_cast<int>(seqs[0].size());
    for (int i = 0; i < n_taxa; ++i)
        if (static_cast<int>(seqs[i].size()) != n_sites)
            throw std::runtime_error("MSA: sequences have unequal length");

    SeqType st = detect_type(seqs);

    std::vector<uint32_t> encoded = encode(seqs, st, n_taxa, n_sites);
    struct DeduplicateResult dedup = deduplicate(encoded, n_taxa, n_sites);

    // Remove fully-undetermined columns (all taxa gap-only) — matches RAxML.
    {
        uint32_t state_mask = (st == SeqType::AA) ? AA_ALL_STATES : (DNA_GAP - 1u);
        std::vector<int> keep;
        for (int p = 0; p < dedup.n_unique; p++) {
            bool any_state = false;
            for (int t = 0; t < n_taxa; t++) {
                if (dedup.patterns[t * dedup.n_unique + p] & state_mask) {
                    any_state = true; break;
                }
            }
            if (any_state) keep.push_back(p);
        }
        if ((int)keep.size() < dedup.n_unique) {
            int nk = (int)keep.size();
            std::vector<uint32_t> new_pat(n_taxa * nk);
            std::vector<int>      new_wts(nk);
            for (int i = 0; i < nk; i++) {
                int p = keep[i];
                new_wts[i] = dedup.weights[p];
                for (int t = 0; t < n_taxa; t++)
                    new_pat[t * nk + i] = dedup.patterns[t * dedup.n_unique + p];
            }
            dedup.patterns = std::move(new_pat);
            dedup.weights  = std::move(new_wts);
            dedup.n_unique = nk;
        }
    }

    struct FilterResult filtered;
    if (parsimony_informative_only) {
        filtered = filter_to_parsimony_informative(dedup.patterns, dedup.weights,
                                                   n_taxa, dedup.n_unique, st);
    } else {
        filtered = {dedup.patterns, dedup.weights, dedup.n_unique};
    }

    MSA msa;
    msa.seq_type = st;
    msa.n_taxa = n_taxa;
    msa.n_patterns = filtered.n_patterns;
    msa.n_sites_orig = n_sites;
    msa.taxon_names = names;
    msa.data = std::move(filtered.patterns);
    msa.weights = std::move(filtered.weights);
    msa.parsimony_offset = filtered.offset;
    return msa;
}

MSA MSA::from_fasta(const std::string& path,
                     bool parsimony_informative_only) {
    return from_fasta(path, {}, parsimony_informative_only);
}

MSA MSA::from_fasta(const std::string& path,
                     const std::vector<std::string>& taxon_order,
                     bool parsimony_informative_only) {
    struct RawMSA raw = read_fasta(path);
    int n_taxa = static_cast<int>(raw.names.size());

    // Reorder taxa if requested
    std::vector<std::string> names = raw.names;
    std::vector<std::string> seqs = raw.seqs;

    if (!taxon_order.empty()) {
        assert(static_cast<int>(taxon_order.size()) == n_taxa);
        std::unordered_map<std::string, int> name_to_row;
        for (int i = 0; i < n_taxa; ++i)
            name_to_row[raw.names[i]] = i;

        names.resize(n_taxa);
        seqs.resize(n_taxa);
        for (int i = 0; i < n_taxa; ++i) {
            auto it = name_to_row.find(taxon_order[i]);
            if (it == name_to_row.end())
                throw std::runtime_error("Unknown taxon in taxon_order: " + taxon_order[i]);
            names[i] = taxon_order[i];
            seqs[i] = raw.seqs[it->second];
        }
    }

    // Encode + dedup + filter + build via the shared path.
    return from_raw(names, seqs, parsimony_informative_only);
}

MSA MSA::from_sequences(const std::vector<std::string>& names,
                        const std::vector<std::string>& seqs,
                        bool parsimony_informative_only) {
    return from_raw(names, seqs, parsimony_informative_only);
}

void MSA::print_summary() const {
    std::cerr << "MSA(type=" << (seq_type == SeqType::DNA ? "DNA" : "AA")
              << ", n_taxa=" << n_taxa
              << ", n_patterns=" << n_patterns
              << ", n_sites_orig=" << n_sites_orig
              << ")\n";
}
