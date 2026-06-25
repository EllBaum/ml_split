#!/usr/bin/env python3
"""In-process batch merge over real MSAs -- scale stress test for the
thread_local-resize fix.

The thread_local-size bug only triggers when several merges of DIFFERENT
pattern counts run in ONE process. A subprocess-per-MSA loop (the safe path)
never exercises it. So each invocation of this driver processes a CHUNK of
MSAs in a single process, and -- to guarantee the trigger ordering -- first
runs one tiny "priming" merge that sizes the thread_local buffers SMALL, so the
subsequent full-size merges would overflow them on un-fixed code.

For each real MSA it builds an induced merge (split the true tree, prune to two
overlapping subtrees with inherited outgroups -- no search_cli needed) and
asserts the merge contract:
  taxa-complete, faithful both sides (RF=0), ll-consistent (score==loglik).

On un-fixed code this segfaults (exit 139) within the chunk; on fixed code it
prints OK for every MSA. Any assertion failure prints the offending index and
exits nonzero.

Usage:
  PYTHONPATH=src python3 python/fuzz_batch.py \
      --tree-dir DIR --msa-dir DIR --start 1 --count 3 [--no-prime]
"""
import argparse, os, random, re, sys, time
import ml_split
from validate_merge import (parse_newick, prune_newick, rf_distance,
                            inherited_clades, choose_split)
import fuzz_merge  # reuse the tiny-case generator for priming


def read_fasta(path):
    seqs, name, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line[0] == ">":
                if name is not None:
                    seqs[name] = "".join(buf)
                name = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


def leaves(nwk):
    return set(re.findall(r"[A-Za-z0-9_.]+(?=:)", nwk))


def one_real_merge(idx, tree_dir, msa_dir, frac_a, frac_b, model, full_blo):
    stem = "random_tree_%04d" % idx
    tpath = os.path.join(tree_dir, stem + ".nwk")
    mpath = os.path.join(msa_dir, stem + ".fasta")
    if not (os.path.exists(tpath) and os.path.exists(mpath)):
        return None  # missing pair -> skip (counted, not failed)

    true = open(tpath).read().strip()
    seqs = read_fasta(mpath)
    root = parse_newick(true)
    taxa = sorted(seqs.keys())

    rng = random.Random(1000 + idx)  # deterministic per index
    realA, realB, inhA, inhB = choose_split(root, taxa, frac_a, frac_b, rng)
    if len(realA) < 3 or len(realB) < 3:
        return None
    O_A = rng.choice(sorted(realB)); O_B = rng.choice(sorted(realA))

    A_taxa = sorted(realA) + sorted(inhA) + [O_A]
    B_taxa = sorted(realB) + sorted(inhB) + [O_B]
    nwkA = prune_newick(root, set(A_taxa))
    nwkB = prune_newick(root, set(B_taxa))
    icA = inherited_clades(nwkA, set(inhA))
    icB = inherited_clades(nwkB, set(inhB))
    subA = {t: seqs[t] for t in A_taxa}
    subB = {t: seqs[t] for t in B_taxa}

    t0 = time.time()
    r = ml_split.merge(newick_a=nwkA, newick_b=nwkB, interest_a=O_A,
                       interest_b=O_B, inherited_a=icA, inherited_b=icB,
                       msa_a=subA, msa_b=subB, model=model, full_blo=full_blo)
    dt = time.time() - t0

    want = set(realA) | set(realB) | set(inhA) | set(inhB)
    got = leaves(r.newick)
    assert got == want, ("taxa", idx, sorted(want ^ got))

    coreA = set(realA) | set(inhA); coreB = set(realB) | set(inhB)
    fA = rf_distance(prune_newick(parse_newick(r.newick), coreA),
                     prune_newick(parse_newick(nwkA), coreA))[0]
    fB = rf_distance(prune_newick(parse_newick(r.newick), coreB),
                     prune_newick(parse_newick(nwkB), coreB))[0]
    assert fA == 0, ("faith_a", idx, fA)
    assert fB == 0, ("faith_b", idx, fB)

    if full_blo == "off":
        # loglik is the reals-only 5-BLO search score
        reals = set(realA) | set(realB)
        stripped = prune_newick(parse_newick(r.newick), reals)
        ll2 = ml_split.score(stripped, {t: seqs[t] for t in reals}, model)
    else:
        # loglik is the optimized LL of the full output tree (all taxa)
        ll2 = ml_split.score(r.newick, {t: seqs[t] for t in want}, model)
    assert abs(ll2 - r.loglik) < 1e-4, ("ll", idx, r.loglik, ll2)

    return (len(want), dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree-dir", required=True)
    ap.add_argument("--msa-dir", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--frac-a", type=float, default=0.4)
    ap.add_argument("--frac-b", type=float, default=0.4)
    ap.add_argument("--full-blo", choices=["off","fast","thorough"], default="off")
    ap.add_argument("--model", default="JC")
    ap.add_argument("--no-prime", action="store_true",
                    help="skip the small priming merge (then the trigger "
                         "depends on chunk pattern-count ordering)")
    args = ap.parse_args()

    # Prime: one tiny merge sizes the thread_local buffers small, so the real
    # full-size merges below would overflow them on un-fixed code.
    if not args.no_prime:
        fuzz_merge.one_case(random.Random(12345))
        sys.stdout.write("primed (small merge done)\n"); sys.stdout.flush()

    ran = skipped = 0
    sizes = []; total_t = 0.0
    for idx in range(args.start, args.start + args.count):
        sys.stdout.write("idx %d ... " % idx); sys.stdout.flush()
        res = one_real_merge(idx, args.tree_dir, args.msa_dir,
                             args.frac_a, args.frac_b, args.model, args.full_blo)
        if res is None:
            skipped += 1; sys.stdout.write("skip\n")
        else:
            n, dt = res; ran += 1; sizes.append(n); total_t += dt
            sys.stdout.write("OK  taxa=%d  %.1fs\n" % (n, dt))
        sys.stdout.flush()

    print("CHUNK DONE  ran=%d skipped=%d  taxa=%s  total=%.1fs"
          % (ran, skipped,
             ("%d-%d" % (min(sizes), max(sizes))) if sizes else "-", total_t))


if __name__ == "__main__":
    main()