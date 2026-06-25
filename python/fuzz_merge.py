#!/usr/bin/env python3
"""Property-based fuzzer for ml_split.merge.

Generates many random (tree, partition, alignment) cases and asserts the merge
CONTRACT on every one -- not a fixed set of hand-picked tests. The partitions
deliberately include heavy, scattered inherited taxa, which is what produces the
co-anchored chains that broke faithfulness before, so the fuzzer hammers that
class directly.

For each case it checks:
  1. taxa-complete : merged taxa == realA u realB u inhA u inhB (interest dropped)
  2. faithful      : RF(merged|sideA, subtreeA)==0 and ==0 for B
  3. ll-consistent : |reported loglik - score(reals-only merged tree)| < 1e-6
  4. deterministic : merging the same inputs twice gives an identical newick
  5. order-invariant: permuting MSA row order AND reversing the inherited-clade
                      list gives the same set of merged bipartitions (guards the
                      set-iteration-order bug that caused the flicker)

Run (from the repo, after building the extension or installing the wheel):
  PYTHONPATH=src python python/fuzz_merge.py --cases 5000 --seed 1
Any failure prints the exact reproducing case (taxa, subtrees, partition) and
exits nonzero.
"""
import argparse, random, sys, re
import ml_split
from validate_merge import (parse_newick, prune_newick, rf_distance,
                            inherited_clades, bipartitions)

DNA = "ACGT"


def random_tree(taxa, rng):
    """Random unrooted binary newick (trifurcating root) with random BLs."""
    def bl(): return '%.3f' % rng.uniform(0.02, 0.4)
    nodes = [t + ':' + bl() for t in taxa]
    rng.shuffle(nodes)
    while len(nodes) > 3:
        a = nodes.pop(rng.randrange(len(nodes)))
        b = nodes.pop(rng.randrange(len(nodes)))
        nodes.append('(' + a + ',' + b + '):' + bl())
    return '(' + ','.join(nodes) + ');'


def random_msa(taxa, rng, length):
    return {t: ''.join(rng.choice(DNA) for _ in range(length)) for t in taxa}


def leaves(nwk):
    return set(re.findall(r'[A-Za-z0-9_]+(?=:)', nwk))


def splits(nwk, restrict):
    """Set of frozenset bipartition-sides of nwk restricted to `restrict`."""
    bp = bipartitions(parse_newick(prune_newick(parse_newick(nwk), restrict)), restrict)
    return {frozenset(min(s, restrict - s, key=lambda x: (len(x), sorted(x)))) for s in bp}


def one_case(rng, dump=False):
    n = rng.randint(8, 34)
    taxa = ['t%02d' % i for i in range(n)]
    true = random_tree(taxa, rng)
    root = parse_newick(true)
    # partition into realA/realB with heavy inherited (scattered -> co-anchored)
    fa = rng.uniform(0.30, 0.48); fb = rng.uniform(0.30, 0.48)
    from validate_merge import choose_split
    realA, realB, inhA, inhB = choose_split(root, taxa, fa, fb, rng)
    if len(realA) < 2 or len(realB) < 2:
        return None                      # degenerate split; skip
    O_A = rng.choice(sorted(realB)); O_B = rng.choice(sorted(realA))
    # The per-side search tree is reals-only (interest + inherited removed), so
    # each side needs >=3 REAL taxa. Smaller is out of the intended domain and
    # the merge rejects it cleanly; skip it here.
    if len(realA) < 3 or len(realB) < 3:
        return None
    seqs = random_msa(taxa, rng, rng.choice([40, 60, 80]))

    A_taxa = sorted(realA) + sorted(inhA) + [O_A]
    B_taxa = sorted(realB) + sorted(inhB) + [O_B]
    nwkA = prune_newick(root, set(A_taxa))
    nwkB = prune_newick(root, set(B_taxa))
    icA = inherited_clades(nwkA, set(inhA))
    icB = inherited_clades(nwkB, set(inhB))
    subA = {t: seqs[t] for t in A_taxa}
    subB = {t: seqs[t] for t in B_taxa}

    def do_merge(ica, icb, sa, sb):
        return ml_split.merge(newick_a=nwkA, newick_b=nwkB, interest_a=O_A,
                              interest_b=O_B, inherited_a=ica, inherited_b=icb,
                              msa_a=sa, msa_b=sb, model="JC")

    if dump:
        print('n        :', n)
        print('subtree A:', nwkA)
        print('subtree B:', nwkB)
        print('O_A,O_B  :', O_A, O_B)
        print('inhA,inhB:', sorted(inhA), sorted(inhB))
        print('cladesA  :', icA)
        print('cladesB  :', icB)
        print('subA keys:', list(subA)); print('subB keys:', list(subB))
        print('--- calling merge ---'); sys.stdout.flush()

    r = do_merge(icA, icB, subA, subB)
    ctx = dict(true=true, nwkA=nwkA, nwkB=nwkB, O_A=O_A, O_B=O_B,
               inhA=sorted(inhA), inhB=sorted(inhB), icA=icA, icB=icB)

    # 1. taxa-complete
    want = set(realA) | set(realB) | set(inhA) | set(inhB)
    got = leaves(r.newick)
    assert got == want, ('taxa', sorted(want ^ got), ctx)

    # 2. faithful both sides
    coreA = set(realA) | set(inhA); coreB = set(realB) | set(inhB)
    fA = rf_distance(prune_newick(parse_newick(r.newick), coreA),
                     prune_newick(parse_newick(nwkA), coreA))[0]
    fB = rf_distance(prune_newick(parse_newick(r.newick), coreB),
                     prune_newick(parse_newick(nwkB), coreB))[0]
    assert fA == 0, ('faith_a', fA, ctx)
    assert fB == 0, ('faith_b', fB, ctx)

    # 3. ll-consistent (reals-only tree scored independently)
    reals = set(realA) | set(realB)
    stripped = prune_newick(parse_newick(r.newick), reals)
    ll2 = ml_split.score(stripped, {t: seqs[t] for t in reals}, "JC")
    assert abs(ll2 - r.loglik) < 1e-6, ('ll', r.loglik, ll2, ctx)

    # 4. deterministic
    r2 = do_merge(icA, icB, subA, subB)
    assert r2.newick == r.newick and r2.loglik == r.loglik, ('determinism', ctx)

    # 5. order-invariant: shuffle MSA dict order + reverse inherited clade lists
    permA = {t: subA[t] for t in rng.sample(list(subA), len(subA))}
    permB = {t: subB[t] for t in rng.sample(list(subB), len(subB))}
    r3 = do_merge(list(reversed(icA)), list(reversed(icB)), permA, permB)
    s_orig = splits(r.newick, want)
    s_perm = splits(r3.newick, want)
    assert s_orig == s_perm, ('order_invariance', ctx)

    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--log-seeds', action='store_true',
                    help='print each case sub-seed before running (find a crash)')
    ap.add_argument('--replay', type=int, default=None,
                    help='run a single case by its sub-seed, dumping inputs first')
    args = ap.parse_args()

    if args.replay is not None:
        print('REPLAY sub-seed', args.replay)
        res = one_case(random.Random(args.replay), dump=True)
        print('case completed ok (res=%s)' % res)
        return

    rng = random.Random(args.seed)
    ran = 0; skipped = 0; sizes = []
    for i in range(args.cases):
        cs = rng.getrandbits(63)
        if args.log_seeds:
            sys.stderr.write('case %d sub-seed %d\n' % (i, cs)); sys.stderr.flush()
        try:
            res = one_case(random.Random(cs))
            if res is None:
                skipped += 1
            else:
                ran += 1; sizes.append(res)
        except AssertionError as e:
            print('\nFAIL on case %d:' % i)
            payload = e.args[0]
            kind = payload[0]
            ctx = payload[-1]
            print('  invariant :', kind)
            print('  detail    :', payload[1:-1])
            print('  true      :', ctx['true'])
            print('  subtree A :', ctx['nwkA'])
            print('  subtree B :', ctx['nwkB'])
            print('  O_A,O_B   :', ctx['O_A'], ctx['O_B'])
            print('  inhA,inhB :', ctx['inhA'], ctx['inhB'])
            print('  cladesA/B :', ctx['icA'], ctx['icB'])
            sys.exit(1)
        if (i + 1) % 1000 == 0:
            print('  %d cases ok (ran=%d skipped=%d, taxa %d-%d)'
                  % (i + 1, ran, skipped, min(sizes), max(sizes)))

    print('\nALL %d CASES PASS  (ran=%d, skipped-degenerate=%d, taxa range %d-%d)'
          % (args.cases, ran, skipped, min(sizes), max(sizes)))
    print('contract held: taxa-complete, faithful both sides, ll-consistent, '
          'deterministic, order-invariant.')


if __name__ == '__main__':
    main()
