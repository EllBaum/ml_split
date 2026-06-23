// counters.h — Instrumentation-only counters. Read-only; no behavior change.
//
// Each counter is a thread-local atomic-incremented uint64_t. Printed once
// at program exit (Counters destructor), in a single block, sorted by name.
//
// All counters live under names like "category.subname" so the output groups
// related counters together.
//
// Usage:
//   COUNTER_INC("clv.toward_memcpy");
//   COUNTER_ADD("nr.iters_distribution", iters);
//   COUNTER_HIST("triplet.sweeps", sweep_count);  // bucketed histogram
//
// To compile this OUT entirely (zero overhead), define COUNTERS_DISABLED.

#pragma once

#ifndef COUNTERS_DISABLED

#include <atomic>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <vector>

class Counters {
public:
    static Counters& instance() { static Counters c; return c; }

    void inc(const std::string& name)              { add(name, 1); }
    void add(const std::string& name, uint64_t v)  {
        std::lock_guard<std::mutex> g(mu_);
        sums_[name] += v;
        counts_[name] += 1;
    }
    // Bucketed histogram. Buckets are 0,1,2,3,5,10,20,30+ — covers NR iters
    // and sweep counts cleanly without parameterization.
    void hist(const std::string& name, uint64_t v) {
        std::lock_guard<std::mutex> g(mu_);
        const char* bucket;
        if      (v == 0)  bucket = "_b00";
        else if (v == 1)  bucket = "_b01";
        else if (v == 2)  bucket = "_b02";
        else if (v == 3)  bucket = "_b03";
        else if (v <= 5)  bucket = "_b05";
        else if (v <= 10) bucket = "_b10";
        else if (v <= 20) bucket = "_b20";
        else              bucket = "_b30plus";
        sums_[name + bucket] += 1;
    }

    ~Counters() {
        if (sums_.empty()) return;
        std::fprintf(stderr, "\n=== counters ===\n");
        // First pass: column widths.
        size_t name_w = 0;
        for (auto& kv : sums_) name_w = std::max(name_w, kv.first.size());
        for (auto& kv : sums_) {
            uint64_t n = counts_.count(kv.first) ? counts_[kv.first] : 0;
            if (n > 0) {
                std::fprintf(stderr, "  %-*s  total=%-12llu calls=%-12llu mean=%.2f\n",
                    (int)name_w, kv.first.c_str(),
                    (unsigned long long)kv.second,
                    (unsigned long long)n,
                    (double)kv.second / n);
            } else {
                std::fprintf(stderr, "  %-*s  count=%llu\n",
                    (int)name_w, kv.first.c_str(),
                    (unsigned long long)kv.second);
            }
        }
    }

private:
    std::map<std::string, uint64_t> sums_;
    std::map<std::string, uint64_t> counts_;
    std::mutex mu_;
};

#define COUNTER_INC(name)        ::Counters::instance().inc(name)
#define COUNTER_ADD(name, val)   ::Counters::instance().add(name, (uint64_t)(val))
#define COUNTER_HIST(name, val)  ::Counters::instance().hist(name, (uint64_t)(val))

#else

#define COUNTER_INC(name)        ((void)0)
#define COUNTER_ADD(name, val)   ((void)0)
#define COUNTER_HIST(name, val)  ((void)0)

#endif
