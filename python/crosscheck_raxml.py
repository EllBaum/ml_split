#!/usr/bin/env python3
"""Cross-check ml_split's likelihood against raxml-ng on the SAME tree.

The merge reports the log-likelihood of the search tree (realA ∪ realB, inherited
removed). This script: runs one merge, strips the inherited back off to recover
that search tree (with its branch lengths), scores it three ways and compares —

  (1) merge's reported loglik
  (2) ml_split.score(search_tree, search_msa)        # our engine, fixed BLs
  (3) raxml-ng --evaluate --opt-branches off          # reference, fixed BLs

(1) and (2) must match exactly (sanity of the strip/rescore). (2) vs (3) is the
real cross-check: same tree, same BLs, same model → the two engines should agree
to numerical tolerance. A constant offset usually means a model/frequency
mismatch (see --model note below), not a bug.

  python crosscheck_raxml.py --msa <MSA> --true-tree <tree> \
      --search-cli ../search_cli --out-dir /tmp/xc \
      --raxml /path/to/raxml-ng [--induced] [--model JTT]

Without --raxml it just reports (1)/(2) and prints the raxml command to run.

Model note: ml_split 'JTT' uses the JTT matrix with JTT's own fixed amino-acid
frequencies. The matching raxml-ng model is 'JTT' (fixed freqs). If you change
ml_split to empirical frequencies, use 'JTT+FC' for raxml-ng instead.
"""
import argparse, os, random, re, subprocess
import ml_split
import validate_merge as vm


def build_merge(seqs, true_nwk, search_cli, out_dir, budget, seed, frac_a,
                frac_b, model, induced):
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)
    root = vm.parse_newick(true_nwk)
    taxa = sorted(seqs)
    realA, realB, inhA, inhB = vm.choose_split(root, taxa, frac_a, frac_b, rng)
    O_A = rng.choice(sorted(realB)); O_B = rng.choice(sorted(realA))
    setA = list(realA) + [O_A] + inhA
    setB = list(realB) + [O_B] + inhB
    subA = {t: seqs[t] for t in setA}; subB = {t: seqs[t] for t in setB}
    if induced:
        nwkA = vm.prune_newick(root, set(setA)); nwkB = vm.prune_newick(root, set(setB))
    else:
        def build(seqset, tag):
            fp = os.path.join(out_dir, tag + '.fas'); vm.write_fasta(fp, seqset)
            op = os.path.join(out_dir, tag + '.nwk')
            cmd = [search_cli, '--msa', fp, '--ml', '--model', model,
                   '--seed', str(seed), '--output', op]
            if budget and budget > 0:
                cmd += ['--budget', str(budget)]
            subprocess.run(cmd, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return open(op).read().strip()
        nwkA = build(subA, 'A'); nwkB = build(subB, 'B')
    res = ml_split.merge(newick_a=nwkA, newick_b=nwkB, interest_a=O_A, interest_b=O_B,
                         inherited_a=vm.inherited_clades(nwkA, set(inhA)),
                         inherited_b=vm.inherited_clades(nwkB, set(inhB)),
                         msa_a=subA, msa_b=subB, model=model)
    search_taxa = set(realA) | set(realB)              # inherited stripped
    return res, search_taxa


def run_raxml(raxml, tree_nwk, msa_fasta, model, prefix):
    # evaluate at fixed BLs and fixed model — pure likelihood of the given tree
    for ext in ('.raxml.log', '.raxml.bestTree', '.raxml.rba', '.raxml.startTree'):
        try: os.remove(prefix + ext)
        except OSError: pass
    cmd = [raxml, '--evaluate', '--msa', msa_fasta, '--tree', tree_nwk,
           '--model', model, '--opt-branches', 'off', '--opt-model', 'off',
           '--threads', '1', '--force', 'perf_threads', '--prefix', prefix]
    out = subprocess.run(cmd, capture_output=True, text=True)
    text = out.stdout + out.stderr
    m = re.search(r'Final LogLikelihood:\s*(-?[\d.]+)', text)
    if not m:
        log = prefix + '.raxml.log'
        if os.path.exists(log):
            m = re.search(r'Final LogLikelihood:\s*(-?[\d.]+)', open(log).read())
    if not m:
        raise RuntimeError('could not parse raxml LL.\n' + text[-1500:])
    return float(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--msa', required=True)
    ap.add_argument('--true-tree', required=True)
    ap.add_argument('--search-cli', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--raxml',
                    default='/groups/pupko/ellabaumer/miniconda3/envs/raxmlng/bin/raxml-ng',
                    help='path to raxml-ng (set to "" to skip the raxml step)')
    ap.add_argument('--model', default='JTT')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--budget', type=int, default=0)
    ap.add_argument('--frac-a', type=float, default=0.45)
    ap.add_argument('--frac-b', type=float, default=0.45)
    ap.add_argument('--induced', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    seqs = vm.read_fasta(args.msa)
    true_nwk = open(args.true_tree).read().strip()
    res, search_taxa = build_merge(seqs, true_nwk, args.search_cli, args.out_dir,
                                   args.budget, args.seed, args.frac_a, args.frac_b,
                                   args.model, args.induced)

    # strip inherited -> search tree (recovers the tree whose LL the merge reported)
    T_s = vm.prune_newick(vm.parse_newick(res.newick), search_taxa)
    tpath = os.path.join(args.out_dir, 'search_tree.nwk')
    open(tpath, 'w').write(T_s)
    M_s = {t: seqs[t] for t in search_taxa}
    mpath = os.path.join(args.out_dir, 'search_msa.fasta')
    vm.write_fasta(mpath, M_s)

    ll_score = ml_split.score(T_s, M_s, args.model)
    print('search taxa            : {}'.format(len(search_taxa)))
    print('(1) merge reported LL  : {:.6f}'.format(res.loglik))
    print('(2) ml_split.score LL  : {:.6f}'.format(ll_score))
    print('    (1)-(2) diff       : {:.2e}  {}'.format(
        abs(res.loglik - ll_score),
        'OK' if abs(res.loglik - ll_score) < 1e-4 else 'MISMATCH'))

    if not args.raxml:
        print('\nNo --raxml given. To finish the cross-check, run:')
        print('  {} --evaluate --msa {} --tree {} --model {} '
              '--opt-branches off --opt-model off --threads 1 '
              '--force perf_threads --prefix {}/xc'.format(
                  'raxml-ng', mpath, tpath, args.model, args.out_dir))
        return

    ll_raxml = run_raxml(args.raxml, tpath, mpath, args.model,
                         os.path.join(args.out_dir, 'xc'))
    print('(3) raxml-ng LL        : {:.6f}'.format(ll_raxml))
    diff = abs(ll_raxml - ll_score)
    print('    (2)-(3) diff       : {:.4f}  {}'.format(
        diff, 'AGREE' if diff < 0.1 else
        ('close (model/freq?)' if diff < max(5.0, 0.001 * abs(ll_score)) else 'MISMATCH')))


if __name__ == '__main__':
    main()
