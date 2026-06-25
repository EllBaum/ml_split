// bench_eval.cpp -- time a PURE single-tree likelihood evaluation.
//
// Builds the scorer ONCE, then times repeated full from-scratch recomputes
// (all CLVs + root LL, buffers reused) -- the same work raxml-ng --evaluate
// --opt-branches off --opt-model off does per tree. Reports:
//   * build+first-eval (cold one-shot, what `score()` / one CLI call costs)
//   * per-eval (warm, amortized over --iters) -- the engine's evaluation speed
//
// Single-threaded. Pin a core and warm the FS cache for a clean number:
//   taskset -c 0 ./bench_eval --msa aln.fasta --tree tree.nwk --model JC --warmup 20 --iters 500
//
// Compare against raxml-ng on the SAME node, SAME alignment+tree:
//   raxml-ng --evaluate --msa aln.fasta --tree tree.nwk --model JC --opt-branches off --opt-model off --threads 1 --force perf_threads
// (see notes printed at the end for getting raxml's PURE per-eval number).
#include "msa.h"
#include "tree.h"
#include "likelihood_scorer.h"
#include "jck.h"
#include "jtt.h"
#include <chrono>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

int main(int argc, char** argv) {
    std::string msa_path, tree_path, model = "JC";
    int iters = 500, warmup = 20;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]{ return std::string(argv[++i]); };
        if      (a == "--msa")    msa_path  = next();
        else if (a == "--tree")   tree_path = next();
        else if (a == "--model")  model     = next();
        else if (a == "--iters")  iters     = std::stoi(next());
        else if (a == "--warmup") warmup    = std::stoi(next());
    }
    if (msa_path.empty() || tree_path.empty()) {
        std::cerr << "usage: bench_eval --msa <fasta> --tree <newick> "
                     "[--model JC|JTT] [--warmup N] [--iters N]\n";
        return 2;
    }

    std::string newick;
    { std::ifstream f(tree_path); std::string line;
      while (std::getline(f, line)) newick += line; }

    // -- Build (allocation + first full eval), timed as the cold one-shot ------
    auto t0 = clk::now();
    std::vector<double> probe(1 << 16, 0.1);
    Tree t_probe = Tree::from_newick(newick, probe.data());
    MSA full = MSA::from_fasta(msa_path, t_probe.taxon_names, false);  // tree-order rows
    std::vector<double> bl((size_t)t_probe.n_nodes * 3, 0.1);
    Tree t = Tree::from_newick(newick, bl.data(), t_probe.taxon_names);
    std::unique_ptr<SubstModel> mdl =
        (model == "JTT") ? std::unique_ptr<SubstModel>(new JTT())
                         : std::unique_ptr<SubstModel>(
                               new JCK(full.seq_type == SeqType::AA ? 20 : 4));
    LikelihoodScorer S(t, full, *mdl, bl.data());
    auto t1 = clk::now();
    double ll = S.score();

    // -- Warm, then time pure recomputes --------------------------------------
    for (int i = 0; i < warmup; ++i) S.recompute_full();
    std::vector<double> per;
    per.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto a = clk::now();
        double v = S.recompute_full();
        auto b = clk::now();
        per.push_back(secs(a, b));
        if (v != ll) { std::cerr << "LL drift: " << v << " vs " << ll << "\n"; return 1; }
    }
    std::sort(per.begin(), per.end());
    double total = 0; for (double x : per) total += x;
    double mean = total / per.size();
    double med  = per[per.size() / 2];
    double best = per.front();

    std::cout.setf(std::ios::fixed);
    std::cout << "MSA            : " << msa_path << "\n"
              << "taxa/patterns  : " << t.n_taxa << " / " << S.n_patterns() << "\n"
              << "model          : " << model
              << " (states=" << S.n_states() << ")\n"
              << "loglik         : " << std::setprecision(6) << ll << "\n"
              << "-----\n"
              << "build+1st eval : " << std::setprecision(3) << secs(t0, t1) * 1e3
              << " ms   (cold one-shot: parse+alloc+evaluate)\n"
              << "per-eval  best : " << best * 1e6 << " us\n"
              << "per-eval  med  : " << med  * 1e6 << " us\n"
              << "per-eval  mean : " << mean * 1e6 << " us   (over " << iters
              << " warm recomputes)\n";
    return 0;
}