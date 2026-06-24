#!/usr/bin/env python3
"""Certify a freshly-built ml_split wheel.

Run this ONCE after building and installing the wheel, BEFORE anyone uses it:

    pip install --force-reinstall dist/ml_split-*.whl --break-system-packages
    python tests/smoke_wheel.py

It imports ml_split from the INSTALLED package (not the source tree), runs a
small merge that includes a CO-ANCHORED inherited chain (ia1,ia2 stacked on one
side), and asserts the merge contract holds:

  * module exposes merge, score, MergeResult
  * every input taxon is present, interest outgroups are NOT dropped
  * faithful: merged restricted to each side == that input subtree (RF == 0)
  * score() on the reals-only merged tree reproduces the reported loglik
  * loglik is finite

The co-anchored case means a STALE wheel (built before the ordering fix) fails
the faithful check, not just a "looks fine" pass. The checks are topological /
relative, so they do NOT false-fail on legitimate numeric differences between
your build and theirs (SIMD, flags). Exit code 0 = PASS, 1 = FAIL.
"""
import os, sys, math

HERE = os.path.dirname(os.path.abspath(__file__))
# Pure-python RF scaffolding (no ml_split inside the parse/RF helpers used here).
sys.path.insert(0, os.path.join(HERE, "..", "python"))
from validate_merge import parse_newick, prune_newick, rf_distance  # noqa: E402

import ml_split  # noqa: E402

fails = []
def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        fails.append(msg)

# ── Where did ml_split come from? ────────────────────────────────────────────
mod_path = os.path.abspath(getattr(ml_split, "__file__", ""))
print("ml_split imported from:", mod_path)
repo_src = os.path.abspath(os.path.join(HERE, "..", "src"))
if mod_path.startswith(repo_src):
    print("  NOTE: this is the repo src/ build, not an installed wheel.")
    print("        For a real wheel test, pip install the wheel and run from")
    print("        a directory outside the repo.")

# ── API surface ──────────────────────────────────────────────────────────────
check(hasattr(ml_split, "merge"), "module exposes merge()")
check(hasattr(ml_split, "score"), "module exposes score()  (new entry point — fails on a stale wheel)")
check(hasattr(ml_split, "MergeResult"), "module exposes MergeResult")

# ── Fixture: DNA / JC, 11 taxa, co-anchored ia1,ia2 on side A ─────────────────
msa = {
    "a0": "ACGTACGTACGTACGTACGT", "a1": "ACGTACGTACGAACGTACGT",
    "a2": "ACGTTCGTACGTACGTACGT", "a3": "ACGTTCGTACGAACGTACGT",
    "b0": "TGCATGCATGCATGCATGCA", "b1": "TGCATGCATGGATGCATGCA",
    "b2": "TGCAAGCATGCATGCATGCA", "b3": "TGCAAGCATGGATGCATGCA",
    "ia1": "ACGTACGTTCGTACGTACGT", "ia2": "ACGTACGTACGTTCGTACGT",
    "ib1": "TGCATGCAAGCATGCATGCA",
}
# O_A = b0 (a real leaf of side B); O_B = a0 (a real leaf of side A).
newick_a = "((((a0:.1,ia1:.1):.1,ia2:.1):.1,a1:.1):.1,(a2:.1,a3:.1):.1,b0:.2);"
newick_b = "((b0:.1,b1:.1):.1,(b2:.1,b3:.1):.1,(ib1:.1,a0:.2):.1);"

r = ml_split.merge(newick_a=newick_a, newick_b=newick_b,
                   interest_a="b0", interest_b="a0", msa=msa,
                   inherited_a=[["ia1"], ["ia2"]], inherited_b=[["ib1"]],
                   model="JC")

# ── Taxa complete (interest outgroups are reals of the partner side; kept) ────
import re
got = set(re.findall(r"[A-Za-z0-9_]+(?=:)", r.newick))
want = {"a0","a1","a2","a3","b0","b1","b2","b3","ia1","ia2","ib1"}
check(got == want, "all 11 taxa present; interest outgroups not dropped "
                   + ("" if got == want else f"(missing {sorted(want-got)} extra {sorted(got-want)})"))

# ── Faithful: merged|side == input subtree (RF == 0) ─────────────────────────
def rf_on(x, y, res):
    return rf_distance(prune_newick(parse_newick(x), res),
                       prune_newick(parse_newick(y), res))[0]
coreA = {"a0","a1","a2","a3","ia1","ia2"}
coreB = {"b0","b1","b2","b3","ib1"}
fa = rf_on(r.newick, newick_a, coreA)
fb = rf_on(r.newick, newick_b, coreB)
check(fa == 0, f"faithful side A: merged|A == subtree A (RF={fa})")
check(fb == 0, f"faithful side B: merged|B == subtree B (RF={fb})")

# ── score() reproduces the reported loglik on the reals-only tree ────────────
reals = {"a0","a1","a2","a3","b0","b1","b2","b3"}
stripped = prune_newick(parse_newick(r.newick), reals)
ll_score = ml_split.score(stripped, {k: msa[k] for k in reals}, "JC")
check(math.isfinite(r.loglik), f"loglik is finite ({r.loglik:.6f})")
check(abs(ll_score - r.loglik) <= 1e-6,
      f"score() reproduces loglik (|diff|={abs(ll_score - r.loglik):.2e})")

print()
if fails:
    print(f"SMOKE TEST FAILED ({len(fails)} check(s)):")
    for m in fails:
        print("   -", m)
    sys.exit(1)
print("SMOKE TEST PASSED — wheel is good to hand off.")
sys.exit(0)
