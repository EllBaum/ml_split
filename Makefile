# Makefile — ml_split (outgroup-based subtree merge)
#
#   make            build all test binaries
#   make test       build + run the test suites
#   make clean      remove build dir and binaries
#
# Sources live in src/, tests in tests/, objects in build/.
# src/tree.cpp and src/merge_prep.cpp are scalar. When the merge scorer
# (LikelihoodScorer + kernels) is pulled into src/, add the AVX-512 flags it uses
# with runtime dispatch: -mavx512f -mavx512cd -mavx512dq -mavx512bw -mavx512vl

CXX      ?= g++
CXXSTD   := -std=c++17
OPT      := -O2
WARN     := -Wall -Wextra
SIMD     := -mavx2 -mfma
INC      := -Isrc
CXXFLAGS := $(CXXSTD) $(OPT) $(WARN) $(SIMD) $(INC)

SRC_DIR  := src
TEST_DIR := tests
BUILD    := build

TESTS := test_pruned test_merge_prep

.PHONY: all test clean

all: $(TESTS)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: $(SRC_DIR)/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/%.o: $(TEST_DIR)/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# header dependencies
$(BUILD)/tree.o:            $(SRC_DIR)/tree.cpp $(SRC_DIR)/tree.h
$(BUILD)/merge_prep.o:      $(SRC_DIR)/merge_prep.cpp $(SRC_DIR)/merge_prep.h $(SRC_DIR)/tree.h
$(BUILD)/test_pruned.o:     $(TEST_DIR)/test_pruned.cpp $(SRC_DIR)/tree.h
$(BUILD)/test_merge_prep.o: $(TEST_DIR)/test_merge_prep.cpp $(SRC_DIR)/merge_prep.h $(SRC_DIR)/tree.h

# link
test_pruned:     $(BUILD)/tree.o $(BUILD)/test_pruned.o
	$(CXX) $(CXXFLAGS) $^ -o $@

test_merge_prep: $(BUILD)/tree.o $(BUILD)/merge_prep.o $(BUILD)/test_merge_prep.o
	$(CXX) $(CXXFLAGS) $^ -o $@

test: $(TESTS)
	@echo "-- test_pruned --";     ./test_pruned
	@echo "-- test_merge_prep --"; ./test_merge_prep

clean:
	rm -rf $(BUILD) $(TESTS)
