#!/usr/bin/env python3
"""Validation harness for ml_split.

For one MSA + its true tree:
  1. split taxa into realA / realB (two disjoint true-tree clades, ~frac each)
     and inherited (the leftover), assigning each inherited taxon to one side;
  2. pick interest outgroups O_A (a realB taxon) and O_B (a realA taxon);
  3. write subA.fas = realA + O_A + inhA, subB.fas = realB + O_B + inhB;
  4. build each subtree with search_cli --ml (stepwise + ML search);
  5. merge with ml_split.merge(msa_a, msa_b, ...);
  6. report RF( merged , true tree ) over the shared taxa.

Usage:
  python validate_merge.py --msa aln.fas --true-tree true.nwk \
      --search-cli /path/to/search_cli --out-dir work [--budget N --seed S \
      --frac-a 0.45 --frac-b 0.45]
"""
import argparse, os, random, subprocess, sys
import ml_split


# ── FASTA ────────────────────────────────────────────────────────────────────
def read_fasta(path):
    seqs, name = {}, None
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line[0] == '>':
            name = line[1:].split()[0]; seqs[name] = []
        elif name is not None:
            seqs[name].append(line)
    return {k: ''.join(v) for k, v in seqs.items()}

def write_fasta(path, seqs):
    with open(path, 'w') as f:
        for n, s in seqs.items():
            f.write('>{}\n{}\n'.format(n, s))


# ── Newick ─────────────────────────────────────────────────────────────────────
def parse_newick(s):
    s = s.strip().rstrip(';'); pos = 0
    def clade():
        nonlocal pos
        node = {'name': None, 'children': [], 'bl': 0.0}
        if s[pos] == '(':
            pos += 1
            while True:
                node['children'].append(clade())
                if s[pos] == ',': pos += 1; continue
                if s[pos] == ')': pos += 1; break
        start = pos
        while pos < len(s) and s[pos] not in ',():': pos += 1
        node['name'] = s[start:pos] or None
        if pos < len(s) and s[pos] == ':':
            pos += 1; bstart = pos
            while pos < len(s) and s[pos] not in ',()': pos += 1
            try: node['bl'] = float(s[bstart:pos])
            except ValueError: node['bl'] = 0.0
        return node
    return clade()

def prune_newick(root, keep):
    """Prune `root` to leaves in `keep`, contracting degree-2 nodes and summing
    branch lengths, then unroot any residual bifurcation. Returns a newick str."""
    def rec(node):
        if not node['children']:
            return dict(node) if node['name'] in keep else None
        kids = [k for k in (rec(c) for c in node['children']) if k is not None]
        if not kids:
            return None
        if len(kids) == 1:                      # contract: fold this node's edge in
            kids[0]['bl'] += node.get('bl', 0.0)
            return kids[0]
        return {'name': None, 'children': kids, 'bl': node.get('bl', 0.0)}
    r = rec(root)
    if r is None or not r['children']:
        raise ValueError('prune kept < 2 leaves')
    if len(r['children']) == 2:                 # unroot a residual bifurcation
        a, b = r['children']
        if a['children']:
            b['bl'] += a['bl']; a['children'].append(b); r = a
        elif b['children']:
            a['bl'] += b['bl']; b['children'].append(a); r = b
    def emit(node):
        if not node['children']:
            return '{}:{:.10g}'.format(node['name'], node['bl'])
        inner = ','.join(emit(c) for c in node['children'])
        return '({}):{:.10g}'.format(inner, node['bl'])
    # root has no parent edge
    return '(' + ','.join(emit(c) for c in r['children']) + ');'

def clade_leaves(node):
    if not node['children']:
        return {node['name']}
    out = set()
    for c in node['children']:
        out |= clade_leaves(c)
    return out

def all_clades(root):
    """All descendant-leaf sets (one per internal node)."""
    out = []
    def rec(node):
        if not node['children']:
            return {node['name']}
        s = set()
        for c in node['children']:
            s |= rec(c)
        out.append(frozenset(s))
        return s
    rec(root)
    return out

def bipartitions(root, taxa):
    taxa = set(taxa); ref = min(taxa); bp = set()
    for cl in all_clades(root):
        cl = frozenset(cl & taxa)
        if 1 < len(cl) < len(taxa) - 1:
            side = cl if ref not in cl else frozenset(taxa - cl)
            bp.add(side)
    return bp

def rf_distance(nwk1, nwk2):
    t1, t2 = parse_newick(nwk1), parse_newick(nwk2)
    taxa = clade_leaves(t1) & clade_leaves(t2)
    b1, b2 = bipartitions(t1, taxa), bipartitions(t2, taxa)
    rf = len(b1 ^ b2)
    denom = 2 * (len(taxa) - 3)
    return rf, (rf / denom if denom > 0 else 0.0), len(taxa)


# ── split ──────────────────────────────────────────────────────────────────────
def choose_split(true_root, taxa, frac_a, frac_b, rng):
    """realA, realB = two disjoint true-tree clades ~frac each; inherited = rest."""
    n = len(taxa)
    tgt_a, tgt_b = frac_a * n, frac_b * n
    clades = [frozenset(c & set(taxa)) for c in all_clades(true_root)]
    clades = [c for c in clades if 2 <= len(c) <= n - 2]

    realA = min(clades, key=lambda c: abs(len(c) - tgt_a))
    # realB: a clade disjoint from realA, size closest to target
    disjoint = [c for c in clades if not (c & realA)]
    if disjoint:
        realB = min(disjoint, key=lambda c: abs(len(c) - tgt_b))
    else:                                   # fallback: greedy from complement
        rest = [t for t in taxa if t not in realA]
        rng.shuffle(rest)
        realB = frozenset(rest[:int(round(tgt_b))])
    inherited = [t for t in taxa if t not in realA and t not in realB]
    return set(realA), set(realB), inherited


def inherited_clades(nwk, inherited_set):
    """Maximal connected subtrees whose leaves are all inherited — the unit the
    merge should preserve. Singletons separated by reals come back as [[t]]."""
    root = parse_newick(nwk); out = []
    def rec(node):
        leaves = clade_leaves(node)
        if leaves <= inherited_set:
            return leaves                     # all-inherited: let parent absorb
        for c in node['children']:            # mixed: emit all-inherited children
            cl = rec(c)
            if cl is not None:
                out.append(sorted(cl))
        return None
    top = rec(root)
    if top is not None:
        out.append(sorted(top))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--msa', required=True)
    ap.add_argument('--true-tree', required=True)
    ap.add_argument('--search-cli', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--budget', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--frac-a', type=float, default=0.45)
    ap.add_argument('--frac-b', type=float, default=0.45)
    ap.add_argument('--model', default='JTT')
    ap.add_argument('--induced', action='store_true',
                    help='build subtrees by pruning the true tree (isolates merge '
                         'correctness from subtree-build quality; expect RF near 0)')
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    seqs = read_fasta(args.msa)
    true_nwk = open(args.true_tree).read().strip()
    true_root = parse_newick(true_nwk)
    taxa = sorted(seqs.keys())
    print('taxa: {}'.format(len(taxa)))

    realA, realB, inherited = choose_split(true_root, taxa, args.frac_a, args.frac_b, rng)
    # assign each inherited taxon to exactly one side
    inhA, inhB = [], []
    for t in inherited:
        (inhA if rng.random() < 0.5 else inhB).append(t)
    # interest outgroups: one real taxon from the other side
    O_A = rng.choice(sorted(realB))     # a B taxon, added to A
    O_B = rng.choice(sorted(realA))     # an A taxon, added to B
    print('realA={} realB={} inherited={} (A:{},B:{}) O_A={} O_B={}'.format(
        len(realA), len(realB), len(inherited), len(inhA), len(inhB), O_A, O_B))

    setA = list(realA) + [O_A] + inhA
    setB = list(realB) + [O_B] + inhB
    subA = {t: seqs[t] for t in setA}
    subB = {t: seqs[t] for t in setB}
    fa = os.path.join(args.out_dir, 'subA.fas'); write_fasta(fa, subA)
    fb = os.path.join(args.out_dir, 'subB.fas'); write_fasta(fb, subB)

    def build(fasta, out):
        cmd = [args.search_cli, '--msa', fasta, '--budget', str(args.budget),
               '--ml', '--model', args.model, '--seed', str(args.seed), '--output', out]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return open(out).read().strip()

    if args.induced:
        nwkA = prune_newick(true_root, set(setA))
        nwkB = prune_newick(true_root, set(setB))
        open(os.path.join(args.out_dir, 'A.nwk'), 'w').write(nwkA)
        open(os.path.join(args.out_dir, 'B.nwk'), 'w').write(nwkB)
        print('induced subtree A ({} leaves), B ({} leaves)'.format(len(setA), len(setB)))
    else:
        nwkA = build(fa, os.path.join(args.out_dir, 'A.nwk'))
        nwkB = build(fb, os.path.join(args.out_dir, 'B.nwk'))
        print('built subtree A ({} leaves), B ({} leaves)'.format(len(setA), len(setB)))

    res = ml_split.merge(
        newick_a=nwkA, newick_b=nwkB,
        interest_a=O_A, interest_b=O_B,
        inherited_a=inherited_clades(nwkA, set(inhA)),
        inherited_b=inherited_clades(nwkB, set(inhB)),
        msa_a=subA, msa_b=subB, model=args.model)
    print('merge loglik = {:.4f}'.format(res.loglik))
    open(os.path.join(args.out_dir, 'merged.nwk'), 'w').write(res.newick)

    merged_taxa = clade_leaves(parse_newick(res.newick))
    print('merged taxa: {} (expected {})'.format(len(merged_taxa), len(taxa)))
    missing = set(taxa) - merged_taxa
    extra = merged_taxa - set(taxa)
    if missing: print('  MISSING:', sorted(missing))
    if extra:   print('  EXTRA  :', sorted(extra))

    rf, nrf, ncommon = rf_distance(res.newick, true_nwk)
    print('RF(merged, true) = {}  normalized = {:.4f}  over {} taxa'.format(rf, nrf, ncommon))

    # Merge faithfulness: the merged tree must preserve each input subtree's
    # internal topology exactly (subtrees are rigid). Compare on each side's
    # core leaves = reals + its inherited, excluding the interest outgroup
    # (which is restored from the other side and legitimately moves).
    def rf_on(nwkX, nwkY, restrict):
        rx = prune_newick(parse_newick(nwkX), restrict)
        ry = prune_newick(parse_newick(nwkY), restrict)
        return rf_distance(rx, ry)[0]
    coreA = set(realA) | set(inhA)
    coreB = set(realB) | set(inhB)
    fa = rf_on(res.newick, nwkA, coreA)
    fb = rf_on(res.newick, nwkB, coreB)
    print('faithfulness  RF(merged|A, A) = {}   RF(merged|B, B) = {}   '
          '(both should be 0)'.format(fa, fb))


if __name__ == '__main__':
    main()
