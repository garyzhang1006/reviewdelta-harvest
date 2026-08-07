#!/usr/bin/env python3
"""
Recall probe: does the zero-churn half survive the numbers the extractor
cannot see?

The headline is an absence claim, and NUM only matches decimals with 1-4
integer digits and 1-3 decimal places. A paper reporting integer BLEU or
accuracy scores zero churn by construction. This probe extracts the two
table-number classes NUM rejects, integers and out-of-range decimals, and
runs the same verbatim survival diff on them for every usable pair.

If papers our extractor scores at zero churn also keep their integer table
numbers, the zero spike is a property of the papers. If they churn there,
it is a property of the regex.

Resumable: appends to recall_probe.jsonl.
Usage: python3 recall_probe.py
"""
import json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diff_arms import fetch_source, TABULAR, SLEEP
from collections import Counter

OUT = 'recall_probe.jsonl'
INT = re.compile(r'(?<![\w.\-])(\d{1,6})(?![\w.])')
BIGDEC = re.compile(r'(?<![\w.])(\d{5,}\.\d+|\d+\.\d{4,})(?![\w])')


def missed_numbers(tex):
    out = Counter()
    for m in TABULAR.finditer(tex):
        blk = m.group(0)
        for n in INT.findall(blk):
            out['i' + n] += 1
        for n in BIGDEC.findall(blk):
            out['d' + n] += 1
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
    # zero-droppers first: they carry the claim under test
    todo = sorted((p for p in ok if p['id'] not in done),
                  key=lambda p: p['dropped_share'] > 0)
    print(f"{len(done)} done; {len(todo)} to fetch", file=sys.stderr)
    with open(OUT, 'a') as fh:
        for i, p in enumerate(todo, 1):
            a = fetch_source(f"{p['id']}v1")
            time.sleep(SLEEP)
            b = fetch_source(f"{p['id']}v{p['v_last']}")
            time.sleep(SLEEP)
            if not a or not b:
                r = {'id': p['id'], 'status': 'source-unavailable'}
            else:
                A, B = missed_numbers(a), missed_numbers(b)
                dropped = sum((A - B).values())
                r = {'id': p['id'], 'arm': p['arm'], 'status': 'ok',
                     'result_share': p['dropped_share'],
                     'n_v1': sum(A.values()), 'dropped': dropped,
                     'dropped_share': round(dropped / max(sum(A.values()), 1), 4)}
            fh.write(json.dumps(r) + '\n')
            fh.flush()
            print(f"[{i}/{len(todo)}] {p['id']} {r['status']} "
                  f"miss-drop={r.get('dropped_share', '-')}", file=sys.stderr)
