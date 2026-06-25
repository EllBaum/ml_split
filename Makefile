# Makefile -- ml_split (outgroup-based subtree merge)
#
#   make            build all test binaries
#   make test       build + run the test suites
#   make clean      remove build dir and binaries
#
# Sources in src/, tests in tests/, objects in build/. The merge scorer is the
# umbrella TU src/likelihood_scorer_unit.cpp (AVX-512, per-function target attrs +
# runtime dispatch). COUNTERS_DISABLED / PROF_DISABLED strip counters + profiler.

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

TESTS := test_pruned test_merge_prep test_scorer_smoke test_merge_hook \
         test_merge_util test_merge_e2e test_inherited_e2e test_inherited_structure test_split_choice test_msa_dict test_coanchored_order

.PHONY: all test clean
all: $(TESTS)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: $(SRC_DIR)/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/%.o: $(TEST_DIR)/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# -- scorer umbrella TU --------------------------------------------------------
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

# -- object header deps --------------------------------------------------------
$(BUILD)/tree.o:          $(SRC_DIR)/tree.cpp $(SRC_DIR)/tree.h
$(BUILD)/merge_prep.o:    $(SRC_DIR)/merge_prep.cpp $(SRC_DIR)/merge_prep.h $(SRC_DIR)/tree.h
$(BUILD)/msa.o:           $(SRC_DIR)/msa.cpp $(SRC_DIR)/msa.h
$(BUILD)/merge_session.o: $(SRC_DIR)/merge_session.cpp $(SRC_DIR)/merge_session.h \
                          $(SRC_DIR)/merge_prep.h $(SCORER_HDRS)

$(BUILD)/test_pruned.o:        $(TEST_DIR)/test_pruned.cpp $(SRC_DIR)/tree.h
$(BUILD)/test_merge_prep.o:    $(TEST_DIR)/test_merge_prep.cpp $(SRC_DIR)/merge_prep.h $(SRC_DIR)/tree.h
$(BUILD)/test_scorer_smoke.o:  $(TEST_DIR)/test_scorer_smoke.cpp $(SCORER_HDRS)
$(BUILD)/test_merge_hook.o:    $(TEST_DIR)/test_merge_hook.cpp $(SCORER_HDRS)
$(BUILD)/test_merge_util.o:    $(TEST_DIR)/test_merge_util.cpp $(SRC_DIR)/merge_session.h
$(BUILD)/test_merge_e2e.o:     $(TEST_DIR)/test_merge_e2e.cpp $(SRC_DIR)/merge_session.h $(SCORER_HDRS)
$(BUILD)/test_inherited_e2e.o: $(TEST_DIR)/test_inherited_e2e.cpp $(SRC_DIR)/merge_session.h $(SCORER_HDRS)
$(BUILD)/test_inherited_structure.o: $(TEST_DIR)/test_inherited_structure.cpp $(SRC_DIR)/merge_session.h $(SCORER_HDRS)
$(BUILD)/test_split_choice.o:  $(TEST_DIR)/test_split_choice.cpp $(SRC_DIR)/merge_session.h $(SRC_DIR)/merge_prep.h $(SCORER_HDRS)
$(BUILD)/test_msa_dict.o:      $(TEST_DIR)/test_msa_dict.cpp $(SRC_DIR)/msa.h
$(BUILD)/test_coanchored_order.o: $(TEST_DIR)/test_coanchored_order.cpp $(SRC_DIR)/merge_session.h $(SCORER_HDRS)

# -- link ----------------------------------------------------------------------
MERGE_OBJ := $(BUILD)/scorer.o $(BUILD)/tree.o $(BUILD)/msa.o \
             $(BUILD)/merge_prep.o $(BUILD)/merge_session.o

test_pruned:        $(BUILD)/tree.o $(BUILD)/test_pruned.o
	$(CXX) $(CXXFLAGS) $^ -o $@
test_merge_prep:    $(BUILD)/tree.o $(BUILD)/merge_prep.o $(BUILD)/test_merge_prep.o
	$(CXX) $(CXXFLAGS) $^ -o $@
test_scorer_smoke:  $(BUILD)/scorer.o $(BUILD)/tree.o $(BUILD)/msa.o $(BUILD)/test_scorer_smoke.o
	$(CXX) $(CXXFLAGS) $^ -o $@
test_merge_hook:    $(BUILD)/scorer.o $(BUILD)/tree.o $(BUILD)/msa.o $(BUILD)/test_merge_hook.o
	$(CXX) $(CXXFLAGS) $^ -o $@
test_merge_util:    $(MERGE_OBJ) $(BUILD)/test_merge_util.o
	$(CXX) $(CXXFLAGS) $^ -o $@
test_merge_e2e:     $(MERGE_OBJ) $(BUILD)/test_merge_e2e.o
	$(CXX) $(CXXFLAGS) $^ -o $@
test_inherited_e2e: $(MERGE_OBJ) $(BUILD)/test_inherited_e2e.o
	$(CXX) $(CXXFLAGS) $^ -o $@
test_inherited_structure: $(MERGE_OBJ) $(BUILD)/test_inherited_structure.o
	$(CXX) $(CXXFLAGS) $^ -o $@
test_split_choice:  $(MERGE_OBJ) $(BUILD)/test_split_choice.o
	$(CXX) $(CXXFLAGS) $^ -o $@
test_msa_dict:      $(BUILD)/msa.o $(BUILD)/test_msa_dict.o
	$(CXX) $(CXXFLAGS) $^ -o $@

test: $(TESTS)
	@echo "-- test_pruned --";       ./test_pruned
	@echo "-- test_merge_prep --";   ./test_merge_prep
	@echo "-- test_scorer_smoke --"; ./test_scorer_smoke
	@echo "-- test_merge_hook --";   ./test_merge_hook
	@echo "-- test_merge_util --";   ./test_merge_util
	@echo "-- test_merge_e2e --";    ./test_merge_e2e
	@echo "-- test_inherited_e2e --"; ./test_inherited_e2e
	@echo "-- test_inherited_structure --"; ./test_inherited_structure
	@echo "-- test_split_choice --";  ./test_split_choice
	@echo "-- test_msa_dict --";     ./test_msa_dict
	@echo "-- test_coanchored_order --"; ./test_coanchored_order

clean:
	rm -rf $(BUILD) $(TESTS)
test_coanchored_order: $(MERGE_OBJ) $(BUILD)/test_coanchored_order.o
	$(CXX) $(CXXFLAGS) $^ -o $@

# -- Pure-evaluation benchmark ------------------------------------------------
# Built with the DELIVERED wheel's flags (-mavx2 global; AVX-512 kernels still
# active via their target() attribute + runtime dispatch), so the number
# reflects what the colleagues actually run. For a peak engine-vs-raxml number
# on AVX-512 nodes, override e.g.:
#   make bench_eval BENCH_SIMD="-mavx2 -mfma -mavx512f -mavx512cd -mavx512dq -mavx512bw -mavx512vl"
BENCH_SIMD ?= -mavx2 -mfma
bench_eval: $(TEST_DIR)/bench_eval.cpp $(SRC_DIR)/likelihood_scorer_unit.cpp \
            $(SRC_DIR)/tree.cpp $(SRC_DIR)/msa.cpp $(SCORER_PIECES) $(SCORER_HDRS) \
            $(SRC_DIR)/msa.h $(SRC_DIR)/tree.h
	$(CXX) $(CXXSTD) $(OPT) $(WARN) $(BENCH_SIMD) $(DEFS) $(INC) \
	    $(TEST_DIR)/bench_eval.cpp $(SRC_DIR)/likelihood_scorer_unit.cpp \
	    $(SRC_DIR)/tree.cpp $(SRC_DIR)/msa.cpp -o $@
