// likelihood_scorer_unit.cpp — umbrella translation unit.
//
// Compiles the model bodies + the split LikelihoodScorer source as ONE TU, so
// the scorer's hot loop can inline/devirtualize the model calls instead of
// jumping across the binary. Include order matters: model bodies first, then
// the scorer pieces in their original top-to-bottom order, so every file-scope
// static/kernel precedes its uses.
#include "jck.cpp"
#include "empirical_aa_model.cpp"
#include "jtt.cpp"
#include "gtr.cpp"
#include "likelihood_scorer_core.cpp"
#include "likelihood_scorer_kernels.cpp"
#include "likelihood_scorer_clv.cpp"
#include "likelihood_scorer_roll.cpp"
#include "likelihood_scorer_commit.cpp"
#include "likelihood_scorer_blo.cpp"
#include "likelihood_scorer_debug.cpp"
