#!/usr/bin/env python3
"""Combine bench_corpus shard CSVs and summarize correctness + timing.

  python bench_aggregate.py --glob 'results_bench/shard_*.csv' \
                            --out results_bench/combined.csv
"""
import argparse, csv, glob, statistics as st


def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return None


def pct(xs, p):
    if not xs: return None
    xs = sorted(xs); k = (len(xs) - 1) * p / 100.0
    lo = int(k); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', required=True)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    rows, seen = [], set()
    for fp in sorted(glob.glob(args.glob)):
        with open(fp) as f:
            for r in csv.DictReader(f):
                k = r.get('msa')
                if k and k not in seen:
                    seen.add(k); rows.append(r)

    if not rows:
        print('no rows matched', args.glob); return

    if args.out:
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    ok   = [r for r in rows if r.get('status') == 'ok']
    err  = [r for r in rows if r.get('status') != 'ok']
    agree = [r for r in ok if r.get('agree') in ('1', 1, True, 'True')]
    disagree = [r for r in ok if r not in agree]

    diffs   = [fnum(r['ll_diff'])     for r in ok if fnum(r['ll_diff'])     is not None]
    pereval = [fnum(r['ml_pereval_us']) for r in ok if fnum(r['ml_pereval_us']) is not None]
    cold    = [fnum(r['ml_cold_ms'])  for r in ok if fnum(r['ml_cold_ms'])  is not None]
    ratio   = [fnum(r['e2e_ratio'])   for r in ok if fnum(r['e2e_ratio'])   is not None]
    npat    = [fnum(r['n_pat'])       for r in ok if fnum(r['n_pat'])       is not None]
    us_site = [pe / np for pe, np in zip(pereval, npat) if np]

    def line(name, xs, unit):
        if not xs:
            print('  %-18s (none)' % name); return
        print('  %-18s mean=%.3f median=%.3f p10=%.3f p90=%.3f min=%.3f max=%.3f %s'
              % (name, st.mean(xs), st.median(xs), pct(xs, 10), pct(xs, 90),
                 min(xs), max(xs), unit))

    print('total MSAs        : %d' % len(rows))
    print('  ok              : %d' % len(ok))
    print('  errors          : %d' % len(err))
    print('CORRECTNESS vs raxml-ng (pure evaluation, fixed BLs)')
    print('  AGREE           : %d/%d' % (len(agree), len(ok)))
    if diffs:
        print('  |dLL|           : mean=%.2e median=%.2e max=%.2e'
              % (st.mean(diffs), st.median(diffs), max(diffs)))
    if disagree:
        print('  DISAGREEMENTS (investigate):')
        for r in disagree[:20]:
            print('    %-20s ll_ml=%s ll_raxml=%s dLL=%s'
                  % (r['msa'], r['ll_ml'], r['ll_raxml'], r['ll_diff']))
    print('TIMING')
    line('ml per-eval (us)',  pereval, 'us   (pure CLV kernel)')
    line('ml per-eval/site',  us_site, 'us/pattern')
    line('ml cold one-shot',  cold,    'ms   (build+1st eval, in-process)')
    line('e2e ratio rax/ml',  ratio,   'x    (full-process wall; raxml startup-heavy)')
    if err:
        print('ERRORS:')
        for r in err[:20]:
            print('    %-20s %s' % (r['msa'], r.get('error', '')))


if __name__ == '__main__':
    main()
