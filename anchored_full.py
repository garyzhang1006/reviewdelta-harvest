#!/usr/bin/env python3
"""Position-anchored churn for the full corpus.

The 26-paper probe showed the two headline claims behave differently under
anchoring: the zero share barely moves while the past-20% share more than
doubles. That was a subsample. This runs the same (table, row, column)
anchoring over every usable pair, so the paper can report both bucket shares
under both matchers, corpus-wide.

Anchoring overstates churn, since restructuring a table counts as total
churn; the document matcher understates it. The pair brackets the truth.

Resumable: appends to anchored_full.jsonl.
Usage: python3 anchored_full.py
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diff_arms as da
from claim_probe import cell_grid

OUT = 'anchored_full.jsonl'

if __name__ == '__main__':
    pairs = [json.loads(l) for l in open('pairs.jsonl')]
    ok = [p for p in pairs if p['status'] == 'ok' and p['n_v1'] > 0]
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                done.add(json.loads(line)['id'])
            except Exception:
                pass
    todo = [p for p in ok if p['id'] not in done]
    print(f"{len(done)} done; {len(todo)} to fetch", file=sys.stderr)
    with open(OUT, 'a') as fh:
        for i, p in enumerate(todo, 1):
            v1 = da.fetch_source(f"{p['id']}v1")
            time.sleep(da.SLEEP)
            vn = da.fetch_source(f"{p['id']}v{p['v_last']}")
            time.sleep(da.SLEEP)
            if not v1 or not vn:
                r = {'id': p['id'], 'status': 'source-unavailable'}
            else:
                g1, gn = cell_grid(v1), cell_grid(vn)
                kept = sum(1 for k, v in g1.items() if gn.get(k) == v)
                r = {'id': p['id'], 'arm': p['arm'], 'status': 'ok',
                     'doc_share': p['dropped_share'],
                     'anchored_total': len(g1), 'anchored_kept': kept,
                     'anchored_share': round(1 - kept / len(g1), 4) if g1 else None}
            fh.write(json.dumps(r) + '\n')
            fh.flush()
            print(f"[{i}/{len(todo)}] {p['id']} {r['status']} "
                  f"anch={r.get('anchored_share', '-')}", file=sys.stderr)
