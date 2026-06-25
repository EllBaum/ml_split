#!/usr/bin/env python3
"""Corpus-wide PURE single-tree-evaluation comparison: ml_split vs raxml-ng.

For every MSA in --msa-dir it finds the matching tree in --tree-dir (by filename
stem) and evaluates that tree's log-likelihood at its own branch lengths under
--model with BOTH engines, no optimization. It records:

  CORRECTNESS  ll_ml, ll_raxml, ll_diff, agree  (the engines must give the same
               number; this is the corpus-scale version of the cross-check)
  TIMING       ml_pereval_us : ml_split warm per-evaluation (pure CLV kernel,
                               buffers reused) -- the engine's evaluation speed
               ml_cold_ms    : ml_split build+first-eval, in-process one-shot
               ml_e2e_s      : ml_split as a fresh process doing ONE eval
               rax_e2e_s     : raxml-ng as a fresh process doing ONE eval
               e2e_ratio     : rax_e2e_s / ml_e2e_s

Both _e2e numbers are full-process wall clocks (best of --repeats), so they are
apples-to-apples "cost to invoke the tool once on this MSA". NOTE: raxml-ng has
heavier fixed startup than the tiny bench_eval binary, so e2e_ratio overstates
the pure-kernel gap; ml_pereval_us is the clean kernel number for ml_split. For
raxml's pure kernel you'd amortize startup (multi-tree --evaluate slope) -- do
that on a sample if you need raxml's per-eval, not the whole corpus.

Rows flush per MSA (timeout-safe); --resume skips done MSAs; shard with
--shard-index/--shard-count for SLURM arrays.

Example:
  python bench_corpus.py \
      --msa-dir /groups/pupko/ellabaumer/sim_trees/MSAs_200 \
      --tree-dir /groups/pupko/ellabaumer/sim_trees/trees_200 \
      --bench ../bench_eval \
      --raxml /groups/pupko/ellabaumer/miniconda3/envs/raxmlng/bin/raxml-ng \
      --model JC --msa-glob '*.fasta' \
      --out-csv results_bench/run1.csv --work-dir results_bench/work
"""
import argparse, csv, glob, json, os, re, subprocess, sys, time, traceback

FIELDS = ['msa', 'status', 'n_taxa', 'n_pat', 'll_ml', 'll_raxml', 'll_diff',
          'agree', 'ml_pereval_us', 'ml_cold_ms', 'ml_e2e_s', 'rax_e2e_s',
          'e2e_ratio', 'model', 'error']

TREE_EXT = ('.fasta', '.fas', '.fa', '.phy', '.phylip', '.nwk', '.tree',
            '.treefile', '.txt', '.newick')


def stem(path, suffix):
    b = os.path.basename(path)
    for ext in TREE_EXT:
        if b.endswith(ext):
            b = b[:-len(ext)]; break
    if suffix and b.endswith(suffix):
        b = b[:-len(suffix)]
    return b


def find_trees(tree_dir, suffix):
    out = {}
    for p in glob.glob(os.path.join(tree_dir, '*')):
        if os.path.isfile(p):
            out.setdefault(stem(p, suffix), p)
    return out


def git_commit(path):
    try:
        return subprocess.check_output(['git', '-C', path, 'rev-parse', 'HEAD'],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def time_proc(cmd, repeats):
    """Run cmd `repeats` times; return (min_wall_seconds, stdout_of_last_run)."""
    best = float('inf'); out = ''
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        dt = time.perf_counter() - t0
        out = r.stdout.decode(errors='replace')
        if r.returncode != 0:
            raise RuntimeError('cmd failed (%d): %s' % (r.returncode,
                               ' '.join(cmd)) + '\n' + out[-500:])
        best = min(best, dt)
    return best, out


# --- bench_eval output parsing -----------------------------------------------
_RE_LL   = re.compile(r'loglik\s*:\s*(-?\d+\.?\d*)')
_RE_MED  = re.compile(r'per-eval\s+med\s*:\s*(-?\d+\.?\d*)\s*us')
_RE_COLD = re.compile(r'build\+1st eval\s*:\s*(-?\d+\.?\d*)\s*ms')
_RE_TAX  = re.compile(r'taxa/patterns\s*:\s*(\d+)\s*/\s*(\d+)')

def run_bench(bench, msa, tree, model, warmup, iters):
    out = subprocess.run([bench, '--msa', msa, '--tree', tree, '--model', model,
                          '--warmup', str(warmup), '--iters', str(iters)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    txt = out.stdout.decode(errors='replace')
    if out.returncode != 0:
        raise RuntimeError('bench_eval failed: ' + txt[-500:])
    ll  = float(_RE_LL.search(txt).group(1))
    med = float(_RE_MED.search(txt).group(1))
    cold = float(_RE_COLD.search(txt).group(1))
    mt = _RE_TAX.search(txt)
    n_taxa, n_pat = (int(mt.group(1)), int(mt.group(2))) if mt else ('', '')
    return ll, med, cold, n_taxa, n_pat


# --- raxml-ng --evaluate (no optimization) -----------------------------------
_RE_RAX_LL = re.compile(r'Final LogLikelihood:\s*(-?\d+\.?\d*)')

def raxml_cmd(raxml, msa, tree, model, prefix):
    return [raxml, '--evaluate', '--msa', msa, '--tree', tree, '--model', model,
            '--opt-branches', 'off', '--opt-model', 'off', '--threads', '1',
            '--force', 'perf_threads', '--prefix', prefix, '--redo']

def parse_rax_ll(txt, prefix):
    m = _RE_RAX_LL.search(txt)
    if m:
        return float(m.group(1))
    log = prefix + '.raxml.log'           # fall back to the log file
    if os.path.exists(log):
        m = _RE_RAX_LL.search(open(log, errors='replace').read())
        if m:
            return float(m.group(1))
    raise RuntimeError('could not parse raxml Final LogLikelihood')

def clean_rax(prefix):
    for ext in ('.raxml.log', '.raxml.bestTree', '.raxml.rba', '.raxml.startTree',
                '.raxml.bestModel', '.raxml.reduced.phy', '.raxml.bestTreeCollapsed'):
        try: os.remove(prefix + ext)
        except OSError: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--msa-dir', required=True)
    ap.add_argument('--tree-dir', required=True)
    ap.add_argument('--bench', required=True, help='path to the bench_eval binary')
    ap.add_argument('--raxml', required=True, help='path to raxml-ng')
    ap.add_argument('--out-csv', required=True)
    ap.add_argument('--work-dir', required=True)
    ap.add_argument('--model', default='JC')
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--iters', type=int, default=200,
                    help='warm recomputes for the per-eval kernel number')
    ap.add_argument('--repeats', type=int, default=3,
                    help='process-timing runs per tool (min is kept)')
    ap.add_argument('--tol', type=float, default=1e-2,
                    help='|ll_ml-ll_raxml| below this counts as agree')
    ap.add_argument('--msa-suffix', default='')
    ap.add_argument('--tree-suffix', default='')
    ap.add_argument('--msa-glob', default='*.fasta')
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--shard-count', type=int, default=1)
    ap.add_argument('--resume', action='store_true')
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

    n_ok = n_fail = n_disagree = 0
    for mp in msas:
        key = stem(mp, args.msa_suffix)
        if key in done:
            continue
        row = {f: '' for f in FIELDS}
        row['msa'] = key; row['model'] = args.model
        tp = trees.get(key)
        prefix = os.path.join(args.work_dir, key + '.xc')
        try:
            if tp is None:
                raise FileNotFoundError('no matching tree for stem ' + key)

            # ml_split: kernel per-eval + ll (single rich run)
            ll_ml, pereval_us, cold_ms, n_taxa, n_pat = run_bench(
                args.bench, mp, tp, args.model, args.warmup, args.iters)
            # ml_split: one-eval process wall (fresh process, best of repeats)
            ml_e2e, _ = time_proc([args.bench, '--msa', mp, '--tree', tp,
                                   '--model', args.model, '--warmup', '0',
                                   '--iters', '1'], args.repeats)
            # raxml: one-eval process wall + ll
            rax_e2e, rtxt = time_proc(raxml_cmd(args.raxml, mp, tp, args.model,
                                                prefix), args.repeats)
            ll_rax = parse_rax_ll(rtxt, prefix)
            clean_rax(prefix)

            diff = abs(ll_ml - ll_rax)
            agree = diff < args.tol
            row.update(dict(
                status='ok', n_taxa=n_taxa, n_pat=n_pat,
                ll_ml='%.6f' % ll_ml, ll_raxml='%.6f' % ll_rax,
                ll_diff='%.3e' % diff, agree=int(agree),
                ml_pereval_us='%.1f' % pereval_us, ml_cold_ms='%.2f' % cold_ms,
                ml_e2e_s='%.4f' % ml_e2e, rax_e2e_s='%.4f' % rax_e2e,
                e2e_ratio='%.2f' % (rax_e2e / ml_e2e) if ml_e2e > 0 else ''))
            n_ok += 1
            if not agree:
                n_disagree += 1
        except Exception as e:
            clean_rax(prefix)
            row['status'] = 'error'
            row['error'] = (str(e) or type(e).__name__).replace('\n', ' ')[:300]
            n_fail += 1
            sys.stderr.write('ERROR %s: %s\n' % (key, row['error']))
        w.writerow(row); fcsv.flush()
        print('[%s] %s  agree=%s dLL=%s  ml=%sus rax_e2e=%ss ratio=%s' % (
            row['status'], key, row.get('agree', ''), row.get('ll_diff', ''),
            row.get('ml_pereval_us', ''), row.get('rax_e2e_s', ''),
            row.get('e2e_ratio', '')))

    fcsv.close()
    print('\nshard %d/%d: ok=%d fail=%d disagree=%d (of %d MSAs)' % (
        args.shard_index, args.shard_count, n_ok, n_fail, n_disagree, len(msas)))
    if n_disagree:
        print('WARNING: %d MSAs disagree with raxml -- investigate.' % n_disagree)


if __name__ == '__main__':
    main()
