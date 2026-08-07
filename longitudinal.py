#!/usr/bin/env python3
"""Stage 9: when does churn happen?

For the papers with at least one intermediate version (v_last >= 3), fetch v1,
v2, and v_last, and split total v1-to-final churn into the part already in
place at v2 and the part that arrives later. If churn lands at v2, it precedes
any plausible camera-ready; if it lands late, the arm null deserves suspicion.

Resumable: appends to longitudinal.jsonl and skips ids already present.
"""
import json, os, sys, time
from diff_arms import fetch_source, result_numbers, SLEEP

OUT = 'longitudinal.jsonl'

if __name__ == '__main__':
    ok = [json.loads(l) for l in open('pairs.jsonl')]
    ok = [r for r in ok if r['status'] == 'ok' and r['n_v1'] > 0 and r['v_last'] >= 3]
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                done.add(json.loads(line)['id'])
            except Exception:
                pass
    todo = [r for r in ok if r['id'] not in done]
    print(f"{len(done)} done, {len(todo)} to go", file=sys.stderr)
    with open(OUT, 'a') as fh:
        for i, p in enumerate(todo, 1):
            v1 = fetch_source(f"{p['id']}v1")
            time.sleep(SLEEP)
            v2 = fetch_source(f"{p['id']}v2")
            time.sleep(SLEEP)
            vn = fetch_source(f"{p['id']}v{p['v_last']}")
            time.sleep(SLEEP)
            if not v1 or not v2 or not vn:
                rec = {'id': p['id'], 'arm': p['arm'], 'status': 'source-unavailable'}
            else:
                A, B, C = result_numbers(v1), result_numbers(v2), result_numbers(vn)
                drop_12 = A - B          # v1 numbers gone by v2
                drop_1n = A - C          # v1 numbers gone by final
                early = sum((drop_12 & drop_1n).values())
                rec = {
                    'id': p['id'], 'arm': p['arm'], 'v_last': p['v_last'],
                    'status': 'ok', 'n_v1': sum(A.values()),
                    'dropped_final': sum(drop_1n.values()),
                    'dropped_by_v2': sum(drop_12.values()),
                    'dropped_early': early,
                }
            fh.write(json.dumps(rec) + '\n')
            fh.flush()
            print(f"[{i}/{len(todo)}] {p['id']} {rec['status']} "
                  f"final={rec.get('dropped_final','-')} early={rec.get('dropped_early','-')}",
                  file=sys.stderr)
