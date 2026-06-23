// msa.h — FASTA parsing, bitmask encoding, pattern dedup, optional parsimony filtering.
//
// Sequence types: DNA (4 states + gap, 5 bits) or AA (20 states + gap, 21 bits).
// Bitmask encoding: each character becomes a uint32 with one or more state bits set.
// Output: uint32[n_taxa * n_patterns], int[n_patterns] weights.

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <cassert>

enum class SeqType : int { DNA = 0, AA = 1 };

// DNA bitmask constants (5 bits: A=0, C=1, G=2, T=3, gap=4)
constexpr uint32_t DNA_A   = 1u << 0;
constexpr uint32_t DNA_C   = 1u << 1;
constexpr uint32_t DNA_G   = 1u << 2;
constexpr uint32_t DNA_T   = 1u << 3;
constexpr uint32_t DNA_GAP = 1u << 4;
constexpr uint32_t DNA_ALL = DNA_A | DNA_C | DNA_G | DNA_T | DNA_GAP;

// AA: 20 standard AAs in alphabetical order (ACDEFGHIKLMNPQRSTVWY), gap=bit 20
constexpr uint32_t AA_GAP_BIT    = 1u << 20;
constexpr uint32_t AA_ALL_STATES = (1u << 20) - 1;  // bits 0-19
constexpr uint32_t AA_ALL        = AA_ALL_STATES | AA_GAP_BIT;

class MSA {
public:
    SeqType     seq_type    = SeqType::DNA;
    int         n_taxa      = 0;
    int         n_patterns  = 0;
    int         n_sites_orig = 0;
    std::vector<std::string> taxon_names;
    std::vector<uint32_t>    data;       // size: n_taxa * n_patterns
    std::vector<int>         weights;    // size: n_patterns
    int                      parsimony_offset = 0; // fixed cost of removed non-informative patterns

    // Access: data[taxon * n_patterns + pattern]
    uint32_t at(int taxon, int pattern) const {
        return data[taxon * n_patterns + pattern];
    }

    // Construction
    // parsimony_informative_only=false (default): keep all sites.
    // parsimony_informative_only=true: drop sites constant across taxa
    //   (zero parsimony cost; still informative for likelihood).
    static MSA from_fasta(const std::string& path,
                          bool parsimony_informative_only = false);
    static MSA from_fasta(const std::string& path,
                          const std::vector<std::string>& taxon_order,
                          bool parsimony_informative_only = false);

    // Build directly from in-memory sequences (e.g. a {name: seq} dict), using
    // the same encode/dedup/filter path as from_fasta. names[i] labels seqs[i].
    static MSA from_sequences(const std::vector<std::string>& names,
                              const std::vector<std::string>& seqs,
                              bool parsimony_informative_only = false);

    // Debug
    void print_summary() const;
};

// Encoding tables (defined in msa.cpp)
uint32_t dna_encode(char ch);
uint32_t aa_encode(char ch);
