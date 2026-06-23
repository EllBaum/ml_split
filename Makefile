# Makefile — ml_split (outgroup-based subtree merge)
#
#   make            build all test binaries
#   make test       build + run the test suites
#   make clean      remove build dir and binaries
#
# Sources live in src/, tests in tests/, objects in build/.
# The merge scorer is the umbrella TU src/likelihood_scorer_unit.cpp, built with
# AVX-512 (per-function target attrs + runtime dispatch). COUNTERS_DISABLED and
# PROF_DISABLED strip the counters/profiler so the library and tests stay quiet.

CXX      ?= g++
CXXSTD   := -std=c++17
OPT      := -O2
WARN     := -Wall -Wextra
SIMD     := -mavx2 -mfma -mavx512f -mavx512cd -mavx512dq -mavx512bw -mavx512vl
DEFS     := -DCOUNTERS_DISABLED -DPROF_DISABLED
INC      := -Isrc
CXXFLAGS := $(CXXSTD) $(OPT) $(WARN) $(SIMD) $(DEFS) $(INC)

SRC_DIR  := src
TEST_DIR := tests
BUILD    := build

TESTS := test_pruned test_merge_prep test_scorer_smoke

.PHONY: all test clean

all: $(TESTS)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: $(SRC_DIR)/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/%.o: $(TEST_DIR)/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ── header dependencies ───────────────────────────────────────────────────────
$(BUILD)/tree.o:            $(SRC_DIR)/tree.cpp $(SRC_DIR)/tree.h
$(BUILD)/merge_prep.o:      $(SRC_DIR)/merge_prep.cpp $(SRC_DIR)/merge_prep.h $(SRC_DIR)/tree.h
$(BUILD)/msa.o:             $(SRC_DIR)/msa.cpp $(SRC_DIR)/msa.h

# Scorer umbrella TU: one object, depends on all its included pieces.
SCORER_PIECES := $(SRC_DIR)/likelihood_scorer_core.cpp $(SRC_DIR)/likelihood_scorer_kernels.cpp \
                 $(SRC_DIR)/likelihood_scorer_clv.cpp $(SRC_DIR)/likelihood_scorer_roll.cpp \
                 $(SRC_DIR)/likelihood_scorer_commit.cpp $(SRC_DIR)/likelihood_scorer_blo.cpp \
                 $(SRC_DIR)/likelihood_scorer_debug.cpp \
                 $(SRC_DIR)/jck.cpp $(SRC_DIR)/jtt.cpp $(SRC_DIR)/gtr.cpp \
                 $(SRC_DIR)/empirical_aa_model.cpp
SCORER_HDRS   := $(SRC_DIR)/likelihood_scorer.h $(SRC_DIR)/subst_model.h \
                 $(SRC_DIR)/jck.h $(SRC_DIR)/jtt.h $(SRC_DIR)/gtr.h \
                 $(SRC_DIR)/empirical_aa_model.h $(SRC_DIR)/counters.h $(SRC_DIR)/prof.h \
                 $(SRC_DIR)/tree.h $(SRC_DIR)/msa.h
$(BUILD)/scorer.o: $(SRC_DIR)/likelihood_scorer_unit.cpp $(SCORER_PIECES) $(SCORER_HDRS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $(SRC_DIR)/likelihood_scorer_unit.cpp -o $@

$(BUILD)/test_pruned.o:        $(TEST_DIR)/test_pruned.cpp $(SRC_DIR)/tree.h
$(BUILD)/test_merge_prep.o:    $(TEST_DIR)/test_merge_prep.cpp $(SRC_DIR)/merge_prep.h $(SRC_DIR)/tree.h
$(BUILD)/test_scorer_smoke.o:  $(TEST_DIR)/test_scorer_smoke.cpp $(SCORER_HDRS)

# ── link ──────────────────────────────────────────────────────────────────────
test_pruned:        $(BUILD)/tree.o $(BUILD)/test_pruned.o
	$(CXX) $(CXXFLAGS) $^ -o $@

test_merge_prep:    $(BUILD)/tree.o $(BUILD)/merge_prep.o $(BUILD)/test_merge_prep.o
	$(CXX) $(CXXFLAGS) $^ -o $@

test_scorer_smoke:  $(BUILD)/scorer.o $(BUILD)/tree.o $(BUILD)/msa.o $(BUILD)/test_scorer_smoke.o
	$(CXX) $(CXXFLAGS) $^ -o $@

test: $(TESTS)
	@echo "-- test_pruned --";       ./test_pruned
	@echo "-- test_merge_prep --";   ./test_merge_prep
	@echo "-- test_scorer_smoke --"; ./test_scorer_smoke

clean:
	rm -rf $(BUILD) $(TESTS)
