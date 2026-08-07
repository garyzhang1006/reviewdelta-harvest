#!/usr/bin/env python3
"""
Non-result baseline: rerun the identical survival diff on the numbers our
extractor deliberately rejects.

The paper's churn rate has no scale without this. If arbitrary decimals in
prose (equation constants, citation years with decimals, section-level
numerics) fail to survive revision at the same rate as result-context
numbers, 13.9% is a property of LaTeX editing, not of results. If results
churn well above the ambient rate, the estimand is measuring something.

Baseline class = NUM-shaped decimals that are neither inside a tabular
environment nor adjacent to \\% / \\pm. Same verbatim multiset diff,
same papers, same versions.

Resumable: appends to baseline.jsonl, reruns skip finished ids.
Usage: python3 baseline.py
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diff_arms import fetch_source, NUM, TABULAR, PCT_PM, SLEEP
from collections import Counter

OUT = 'baseline.jsonl'


def nonresult_numbers(tex):
    spans = [m.span() for m in TABULAR.finditer(tex)]
    pct_starts = {m.start(1) for m in PCT_PM.finditer(tex)}
    out = Counter()
    for m in NUM.finditer(tex):
        if m.start(1) in pct_starts:
            continue
        if any(a <= m.start(1) < b for a, b in spans):
            continue
        out[m.group(1)] += 1
    return out


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
            a = fetch_source(f"{p['id']}v1")
            time.sleep(SLEEP)
            b = fetch_source(f"{p['id']}v{p['v_last']}")
            time.sleep(SLEEP)
            if not a or not b:
                r = {'id': p['id'], 'arm': p['arm'], 'status': 'source-unavailable'}
            else:
                A, B = nonresult_numbers(a), nonresult_numbers(b)
                dropped = sum((A - B).values())
                r = {'id': p['id'], 'arm': p['arm'], 'status': 'ok',
                     'n_v1': sum(A.values()),
                     'dropped': dropped,
                     'dropped_share': round(dropped / max(sum(A.values()), 1), 4)}
            fh.write(json.dumps(r) + '\n')
            fh.flush()
            print(f"[{i}/{len(todo)}] {p['id']} {r['status']} "
                  f"drop={r.get('dropped_share', '-')}", file=sys.stderr)
