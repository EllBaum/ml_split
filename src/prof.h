#pragma once
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <cstdint>
#include <vector>

class Prof {
public:
    static Prof& instance() { static Prof p; return p; }

    void add(const std::string& name, double seconds) {
        std::lock_guard<std::mutex> g(mu);
        time_us[name] += (uint64_t)(seconds * 1e6);
        calls[name]   += 1;
    }

    ~Prof() {
        if (time_us.empty()) return;
        std::fprintf(stderr, "\n=== profile (CPU-elapsed in scope) ===\n");
        // Sort by total time descending.
        std::vector<std::pair<std::string,uint64_t>> v(time_us.begin(), time_us.end());
        std::sort(v.begin(), v.end(),
                  [](auto& a, auto& b){return a.second > b.second;});
        for (auto& kv : v) {
            uint64_t c = calls[kv.first];
            std::fprintf(stderr, "  %-40s %10.3f s  %12llu calls  %10.3f us/call\n",
                kv.first.c_str(), kv.second/1e6, (unsigned long long)c,
                c ? (double)kv.second / c : 0.0);
        }
    }
private:
    std::map<std::string, uint64_t> time_us;
    std::map<std::string, uint64_t> calls;
    std::mutex mu;
};

class ProfTimer {
public:
    ProfTimer(const char* n)
        : name(n), t0(std::chrono::steady_clock::now()) {
        depth_for(name)++;
    }
    ~ProfTimer() {
        int& d = depth_for(name);
        if (--d == 0) {
            // Only record the outermost (exclusive) time per name.
            auto dur = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0).count();
            Prof::instance().add(name, dur);
        }
    }
private:
    static int& depth_for(const char* n) {
        // Per-thread, per-name depth counter. Use the pointer as identity
        // (caller passes static string literals, so pointer equality is fine).
        thread_local std::map<const char*, int> tab;
        return tab[n];
    }
    const char* name;
    std::chrono::steady_clock::time_point t0;
};

#define PROF_CONCAT2(a,b) a##b
#define PROF_CONCAT(a,b)  PROF_CONCAT2(a,b)
#ifdef PROF_DISABLED
#define PROF_SCOPE(n)     ((void)0)
#else
#define PROF_SCOPE(n)     ProfTimer PROF_CONCAT(_pt_,__LINE__)(n)
#endif
