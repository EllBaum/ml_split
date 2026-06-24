#!/usr/bin/env python3
"""Combine corpus shard CSVs into one file and print a summary.

  python aggregate_corpus.py --glob 'results_merge/shard_*.csv' \
         --out results_merge/combined.csv
"""
import argparse, csv, glob, statistics as st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    rows, seen, header = [], set(), None
    for fp in sorted(glob.glob(args.glob)):
        with open(fp) as f:
            r = csv.DictReader(f)
            header = r.fieldnames
            for row in r:
                if row['msa'] in seen:
                    continue
                seen.add(row['msa']); rows.append(row)

    if not rows:
        print('no rows matched', args.glob); return
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header); w.writeheader(); w.writerows(rows)

    ok = [r for r in rows if r['status'] == 'ok']
    err = [r for r in rows if r['status'] != 'ok']
    unfaithful = [r for r in ok if r['faithful'] != '1']
    taxa_bad = [r for r in ok if r['taxa_ok'] != '1']
    rf = [float(r['rf_true']) for r in ok if r['rf_true'] not in ('', None)]
    nrf = [float(r['nrf_true']) for r in ok if r['nrf_true'] not in ('', None)]
    tm = [float(r['time_sec']) for r in ok if r['time_sec'] not in ('', None)]

    def s(x):
        return ('mean={:.3f} median={:.3f} min={:.3f} max={:.3f}'.format(
            st.mean(x), st.median(x), min(x), max(x)) if x else 'n/a')

    print('total MSAs        : {}'.format(len(rows)))
    print('  ok              : {}'.format(len(ok)))
    print('  errors          : {}'.format(len(err)))
    print('FAITHFUL merges    : {}/{}  ({} NOT faithful)'.format(
        len(ok) - len(unfaithful), len(ok), len(unfaithful)))
    print('taxa complete      : {}/{}  ({} with missing/extra taxa)'.format(
        len(ok) - len(taxa_bad), len(ok), len(taxa_bad)))
    print('RF(merged,true)    : {}'.format(s(rf)))
    print('normalized RF      : {}'.format(s(nrf)))
    print('time per MSA (s)   : {}'.format(s(tm)))

    xc = [r for r in ok if r.get('ll_raxml') not in ('', None)]
    if xc:
        diff = [float(r['ll_diff']) for r in xc if r['ll_diff'] not in ('', None)]
        agree = [r for r in xc if r.get('raxml_agree') == '1']
        print('raxml cross-check  : {}/{} AGREE (|ΔLL|<0.1)   ΔLL {}'.format(
            len(agree), len(xc), s(diff)))
        bad = [r for r in xc if r.get('raxml_agree') != '1']
        if bad:
            print('  DISAGREE:')
            for r in bad[:20]:
                print('    {}  ml_split={} raxml={} diff={}'.format(
                    r['msa'], r.get('ll_score'), r.get('ll_raxml'), r.get('ll_diff')))
        rerr = [r for r in ok if r.get('raxml_err')]
        if rerr:
            print('  raxml errored on {} MSAs (e.g. {}: {})'.format(
                len(rerr), rerr[0]['msa'], rerr[0]['raxml_err'][:120]))
    print('combined -> {}'.format(args.out))

    if unfaithful:
        print('\nNOT-FAITHFUL (investigate):')
        for r in unfaithful[:20]:
            print('  {}  faith_a={} faith_b={}'.format(r['msa'], r['faith_a'], r['faith_b']))
    if err:
        print('\nERRORS:')
        for r in err[:20]:
            print('  {}  {}'.format(r['msa'], r['error']))


if __name__ == '__main__':
    main()
