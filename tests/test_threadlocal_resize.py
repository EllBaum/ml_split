#!/usr/bin/env python3
"""Minimal reproducer for the thread_local-size bug.
Two merges in ONE process: first with a short alignment, then a long one.
If the contrib/clv temps keep the first call's size, the second overflows.
"""
import sys, random
sys.path.insert(0, 'src')
sys.path.insert(0, 'python')
import ml_split
from validate_merge import parse_newick, prune_newick, inherited_clades, choose_split

rng = random.Random(42)
taxa = ['t%02d' % i for i in range(16)]

def tree():
    def bl(): return '%.3f' % rng.uniform(0.05, 0.3)
    nodes = [t + ':' + bl() for t in taxa]; rng.shuffle(nodes)
    while len(nodes) > 3:
        a = nodes.pop(rng.randrange(len(nodes))); b = nodes.pop(rng.randrange(len(nodes)))
        nodes.append('(' + a + ',' + b + '):' + bl())
    return '(' + ','.join(nodes) + ');'

true = tree(); root = parse_newick(true)
realA, realB, inhA, inhB = choose_split(root, taxa, 0.4, 0.4, rng)
O_A = sorted(realB)[0]; O_B = sorted(realA)[0]
A_taxa = sorted(realA) + sorted(inhA) + [O_A]
B_taxa = sorted(realB) + sorted(inhB) + [O_B]
nwkA = prune_newick(root, set(A_taxa)); nwkB = prune_newick(root, set(B_taxa))
icA = inherited_clades(nwkA, set(inhA)); icB = inherited_clades(nwkB, set(inhB))

def do(L, tag):
    seqs = {t: ''.join(rng.choice('ACGT') for _ in range(L)) for t in taxa}
    sa = {t: seqs[t] for t in A_taxa}; sb = {t: seqs[t] for t in B_taxa}
    r = ml_split.merge(newick_a=nwkA, newick_b=nwkB, interest_a=O_A, interest_b=O_B,
                       inherited_a=icA, inherited_b=icB, msa_a=sa, msa_b=sb, model="JC")
    print('%s: L=%d loglik=%.4f' % (tag, L, r.loglik)); sys.stdout.flush()

do(40,  'merge 1 (short)')
do(200, 'merge 2 (long) ')   # needs a bigger KL than merge 1 sized the temps for
print('both merges completed')
