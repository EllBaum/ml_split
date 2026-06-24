#!/usr/bin/env python3
"""Run the ml_split validation across a corpus of MSAs and save results.

For every MSA in --msa-dir it finds the matching true tree in --tree-dir (by
filename stem, after stripping extensions and an optional suffix), runs one
split→build→merge→compare, and appends a row to --out-csv. Rows are flushed
per MSA so a timeout/preemption never loses completed work; re-running with
--resume skips MSAs already in the CSV. A sidecar <out-csv>.config.json records
the exact run settings (and git commit, if available) for later comparison.

Example:
  python run_corpus.py \
      --msa-dir /groups/pupko/ellabaumer/sim_trees/MSAs_200 \
      --tree-dir /groups/pupko/ellabaumer/sim_trees/trees_200 \
      --search-cli ../search_cli \
      --out-csv results_merge/run1.csv --work-dir results_merge/work \
      --budget 2000 --seed 42
A shard of the corpus (for SLURM arrays): --shard-index K --shard-count N.
"""
import argparse, csv, glob, json, os, subprocess, sys, traceback
import validate_merge as vm

FIELDS = ['msa', 'status', 'n_taxa', 'n_realA', 'n_realB', 'n_inherited',
          'n_inhA', 'n_inhB', 'O_A', 'O_B', 'merged_taxa', 'taxa_ok', 'loglik',
          'rf_true', 'nrf_true', 'n_common', 'faith_a', 'faith_b', 'faithful',
          'll_score', 'll_raxml', 'll_diff', 'raxml_agree', 'raxml_err',
          'mode', 'seed', 'budget', 'time_sec', 'error']


def stem(path, suffix):
    b = os.path.basename(path)
    for ext in ('.fasta', '.fas', '.fa', '.phy', '.phylip', '.nwk', '.tree',
                '.treefile', '.txt', '.newick'):
        if b.endswith(ext):
            b = b[:-len(ext)]; break
    if suffix and b.endswith(suffix):
        b = b[:-len(suffix)]
    return b


def find_trees(tree_dir, tree_suffix):
    out = {}
    for p in glob.glob(os.path.join(tree_dir, '*')):
        if os.path.isfile(p):
            out.setdefault(stem(p, tree_suffix), p)
    return out


def git_commit(path):
    try:
        return subprocess.check_output(['git', '-C', path, 'rev-parse', 'HEAD'],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--msa-dir', required=True)
    ap.add_argument('--tree-dir', required=True)
    ap.add_argument('--search-cli', required=True)
    ap.add_argument('--out-csv', required=True)
    ap.add_argument('--work-dir', required=True)
    ap.add_argument('--budget', type=int, default=0,
                    help='ML search is unlimited by default (0); >0 caps it)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--frac-a', type=float, default=0.45)
    ap.add_argument('--frac-b', type=float, default=0.45)
    ap.add_argument('--model', default='JTT')
    ap.add_argument('--induced', action='store_true')
    ap.add_argument('--raxml', default='',
                    help='path to raxml-ng; if set, cross-check each merge LL')
    ap.add_argument('--msa-suffix', default='', help='e.g. _TRUE to strip from MSA stems')
    ap.add_argument('--tree-suffix', default='')
    ap.add_argument('--msa-glob', default='*.fas')
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--shard-count', type=int, default=1)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--keep-trees', action='store_true',
                    help='keep per-MSA subtree/merged newicks (default: overwrite)')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)

    msas = sorted(glob.glob(os.path.join(args.msa_dir, args.msa_glob)))
    msas = [m for i, m in enumerate(msas) if i % args.shard_count == args.shard_index]
    trees = find_trees(args.tree_dir, args.tree_suffix)

    done = set()
    if args.resume and os.path.exists(args.out_csv):
        with open(args.out_csv) as f:
            for row in csv.DictReader(f):
                done.add(row['msa'])

    # sidecar config (write once)
    cfg = {k: getattr(args, k) for k in vars(args)}
    cfg['git_commit'] = git_commit(os.path.dirname(os.path.abspath(__file__)))
    cfg['n_msas_in_shard'] = len(msas)
    with open(args.out_csv + '.config.json', 'w') as f:
        json.dump(cfg, f, indent=2)

    new_file = not (args.resume and os.path.exists(args.out_csv))
    fcsv = open(args.out_csv, 'a', newline='')
    w = csv.DictWriter(fcsv, fieldnames=FIELDS)
    if new_file:
        w.writeheader(); fcsv.flush()

    n_ok = n_fail = n_unfaithful = 0
    for mp in msas:
        key = stem(mp, args.msa_suffix)
        if key in done:
            continue
        row = {f: '' for f in FIELDS}
        row['msa'] = key
        tp = trees.get(key)
        try:
            if tp is None:
                raise FileNotFoundError('no matching true tree for stem ' + key)
            seqs = vm.read_fasta(mp)
            true_nwk = open(tp).read().strip()
            wd = os.path.join(args.work_dir, key)
            rec = vm.run_one(seqs, true_nwk, args.search_cli, wd,
                             budget=args.budget, seed=args.seed,
                             frac_a=args.frac_a, frac_b=args.frac_b,
                             model=args.model, induced=args.induced,
                             raxml=(args.raxml or None))
            row.update(rec); row['status'] = 'ok'
            n_ok += 1
            if not rec['faithful']:
                n_unfaithful += 1
            if not args.keep_trees:
                for fn in ('A.nwk', 'B.nwk', 'merged.nwk', 'A.fas', 'B.fas',
                           'search_tree.nwk', 'search_msa.fasta',
                           'xc.raxml.log', 'xc.raxml.bestTree', 'xc.raxml.rba',
                           'xc.raxml.startTree', 'xc.raxml.bestModel'):
                    try: os.remove(os.path.join(wd, fn))
                    except OSError: pass
        except Exception as e:
            row['status'] = 'error'
            row['error'] = (str(e) or type(e).__name__).replace('\n', ' ')[:300]
            n_fail += 1
            sys.stderr.write('ERROR {}: {}\n'.format(key, row['error']))
            traceback.print_exc()
        w.writerow(row); fcsv.flush()
        print('[{}] {}  faithful={} rf_true={}'.format(
            row['status'], key, row.get('faithful', ''), row.get('rf_true', '')))

    fcsv.close()
    print('\nshard {}/{}: ok={} fail={} unfaithful={} (out of {} MSAs)'.format(
        args.shard_index, args.shard_count, n_ok, n_fail, n_unfaithful, len(msas)))
    if n_unfaithful:
        print('WARNING: {} merges were not faithful — investigate.'.format(n_unfaithful))


if __name__ == '__main__':
    main()
