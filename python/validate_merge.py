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
    """Even divide along ONE real edge. Pick the true-tree bipartition closest to
    the target ratio (≈50/50 for 45/45), then move ~(1-frac_a-frac_b) of the taxa
    into the inherited bucket, split between the two sides — each inherited taxon
    keeps its native side, so it has a true position the merge can recover.
    Returns realA, realB, inhA, inhB. realA | realB is a real bipartition of the
    true tree (restricted to the reals)."""
    n = len(taxa); taxset = set(taxa)
    clades = [frozenset(c & taxset) for c in all_clades(true_root)]
    clades = [c for c in clades if 2 <= len(c) <= n - 2]

    target = frac_a / (frac_a + frac_b) * n            # ≈ n/2
    G1 = set(min(clades, key=lambda c: abs(len(c) - target)))
    G2 = taxset - G1

    n_inh = max(0, int(round((1.0 - frac_a - frac_b) * n)))
    n_inh1 = int(round(n_inh * len(G1) / n))
    n_inh2 = n_inh - n_inh1
    n_inh1 = min(n_inh1, max(0, len(G1) - 2))          # keep ≥2 reals per side
    n_inh2 = min(n_inh2, max(0, len(G2) - 2))

    g1 = sorted(G1); g2 = sorted(G2)
    rng.shuffle(g1); rng.shuffle(g2)
    inhA = g1[:n_inh1]; inhB = g2[:n_inh2]
    realA = G1 - set(inhA); realB = G2 - set(inhB)
    return realA, realB, inhA, inhB


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
    ap.add_argument('--budget', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--frac-a', type=float, default=0.45)
    ap.add_argument('--frac-b', type=float, default=0.45)
    ap.add_argument('--model', default='JTT')
    ap.add_argument('--induced', action='store_true',
                    help='build subtrees by pruning the true tree (isolates merge '
                         'correctness from subtree-build quality; expect RF near 0)')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seqs = read_fasta(args.msa)
    true_nwk = open(args.true_tree).read().strip()
    rec = run_one(seqs, true_nwk, args.search_cli, args.out_dir,
                  budget=args.budget, seed=args.seed, frac_a=args.frac_a,
                  frac_b=args.frac_b, model=args.model, induced=args.induced,
                  verbose=True)
    for k, v in rec.items():
        print('  {:18s} {}'.format(k, v))


def run_one(seqs, true_nwk, search_cli, out_dir, budget=200, seed=42,
            frac_a=0.45, frac_b=0.45, model='JTT', induced=False, verbose=False,
            raxml=None):
    """Run one split→build→merge→compare. Returns a metrics dict. Raises on
    hard failure (caller records it); merge correctness is reported via the
    faith_a / faith_b fields (must be 0)."""
    import time
    t0 = time.time()
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)
    true_root = parse_newick(true_nwk)
    taxa = sorted(seqs.keys())

    realA, realB, inhA, inhB = choose_split(true_root, taxa, frac_a, frac_b, rng)
    inherited = inhA + inhB
    O_A = rng.choice(sorted(realB))     # a realB taxon, borrowed into A
    O_B = rng.choice(sorted(realA))     # a realA taxon, borrowed into B

    setA = sorted(realA) + [O_A] + sorted(inhA)
    setB = sorted(realB) + [O_B] + sorted(inhB)
    subA = {t: seqs[t] for t in setA}
    subB = {t: seqs[t] for t in setB}

    if induced:
        nwkA = prune_newick(true_root, set(setA))
        nwkB = prune_newick(true_root, set(setB))
    else:
        def build(seqset, tag):
            fp = os.path.join(out_dir, tag + '.fas'); write_fasta(fp, seqset)
            op = os.path.join(out_dir, tag + '.nwk')
            cmd = [search_cli, '--msa', fp, '--ml', '--model', model,
                   '--seed', str(seed), '--output', op]
            if budget and budget > 0:        # else search_cli runs ML unlimited
                cmd += ['--budget', str(budget)]
            subprocess.run(cmd, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return open(op).read().strip()
        nwkA = build(subA, 'A'); nwkB = build(subB, 'B')
    open(os.path.join(out_dir, 'A.nwk'), 'w').write(nwkA)
    open(os.path.join(out_dir, 'B.nwk'), 'w').write(nwkB)

    res = ml_split.merge(
        newick_a=nwkA, newick_b=nwkB, interest_a=O_A, interest_b=O_B,
        inherited_a=inherited_clades(nwkA, set(inhA)),
        inherited_b=inherited_clades(nwkB, set(inhB)),
        msa_a=subA, msa_b=subB, model=model)
    open(os.path.join(out_dir, 'merged.nwk'), 'w').write(res.newick)

    merged_taxa = clade_leaves(parse_newick(res.newick))
    rf, nrf, ncommon = rf_distance(res.newick, true_nwk)

    def rf_on(nwkX, nwkY, restrict):
        return rf_distance(prune_newick(parse_newick(nwkX), restrict),
                           prune_newick(parse_newick(nwkY), restrict))[0]
    faith_a = rf_on(res.newick, nwkA, set(realA) | set(inhA))
    faith_b = rf_on(res.newick, nwkB, set(realB) | set(inhB))

    rec = {
        'n_taxa': len(taxa),
        'n_realA': len(realA), 'n_realB': len(realB),
        'n_inherited': len(inherited), 'n_inhA': len(inhA), 'n_inhB': len(inhB),
        'O_A': O_A, 'O_B': O_B,
        'merged_taxa': len(merged_taxa),
        'taxa_ok': int(merged_taxa == set(taxa)),
        'loglik': round(res.loglik, 4),
        'rf_true': rf, 'nrf_true': round(nrf, 4), 'n_common': ncommon,
        'faith_a': faith_a, 'faith_b': faith_b,
        'faithful': int(faith_a == 0 and faith_b == 0),
        'mode': 'induced' if induced else 'ml',
        'seed': seed, 'budget': budget,
        'time_sec': round(time.time() - t0, 2),
    }

    if raxml:
        # Strip inherited -> search tree (the tree whose LL the merge reported),
        # then score it with ml_split and raxml-ng at the SAME fixed BLs/model.
        search_taxa = set(realA) | set(realB)
        T_s = prune_newick(parse_newick(res.newick), search_taxa)
        tpath = os.path.join(out_dir, 'search_tree.nwk'); open(tpath, 'w').write(T_s)
        M_s = {t: seqs[t] for t in search_taxa}
        mpath = os.path.join(out_dir, 'search_msa.fasta'); write_fasta(mpath, M_s)
        rec['ll_score'] = round(ml_split.score(T_s, M_s, model), 4)
        try:
            ll_raxml = run_raxml(raxml, tpath, mpath, model,
                                 os.path.join(out_dir, 'xc'))
            rec['ll_raxml'] = round(ll_raxml, 4)
            rec['ll_diff'] = round(abs(ll_raxml - rec['ll_score']), 4)
            rec['raxml_agree'] = int(rec['ll_diff'] < 0.1)
        except Exception as e:
            rec['ll_raxml'] = ''
            rec['ll_diff'] = ''
            rec['raxml_agree'] = ''
            rec['raxml_err'] = (str(e) or type(e).__name__).replace('\n', ' ')[:200]
    return rec


def run_raxml(raxml, tree_nwk, msa_fasta, model, prefix):
    """raxml-ng --evaluate at fixed BLs and fixed model: pure LL of the given
    tree. Returns the Final LogLikelihood."""
    import re, subprocess
    for ext in ('.raxml.log', '.raxml.bestTree', '.raxml.rba', '.raxml.startTree',
                '.raxml.bestModel', '.raxml.reduced.phy'):
        try: os.remove(prefix + ext)
        except OSError: pass
    cmd = [raxml, '--evaluate', '--msa', msa_fasta, '--tree', tree_nwk,
           '--model', model, '--opt-branches', 'off', '--opt-model', 'off',
           '--threads', '1', '--force', 'perf_threads', '--prefix', prefix]
    out = subprocess.run(cmd, capture_output=True, text=True)
    text = out.stdout + out.stderr
    m = re.search(r'Final LogLikelihood:\s*(-?[\d.]+)', text)
    if not m and os.path.exists(prefix + '.raxml.log'):
        m = re.search(r'Final LogLikelihood:\s*(-?[\d.]+)',
                      open(prefix + '.raxml.log').read())
    if not m:
        raise RuntimeError('could not parse raxml LL: ' + text[-400:])
    return float(m.group(1))


if __name__ == '__main__':
    main()
