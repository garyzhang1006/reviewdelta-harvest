#!/usr/bin/env python3
"""
Stage 7: rerun the diff counting only numbers matched by the percent/plus-minus
rule, which hand-labeling found 100% precise (n=19), against the tabular rule's
79.4%.

If the arm null and the bimodal shape both survive on the clean subset, neither
is an artifact of the extractor swallowing hyperparameter tables.

Resumable: appends to pairs_clean.jsonl.
"""
import json, os, sys, time
from diff_arms import fetch_source, PCT_PM, SLEEP
from collections import Counter

OUT = 'pairs_clean.jsonl'


def clean_numbers(tex):
    return Counter(m.group(1) for m in PCT_PM.finditer(tex))


if __name__ == '__main__':
    want = [json.loads(l) for l in open('pairs.jsonl')]
    want = [p for p in want if p['status'] == 'ok']
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                done.add(json.loads(line)['id'])
            except Exception:
                pass
    todo = [p for p in want if p['id'] not in done]
    print(f"{len(done)} done, {len(todo)} to go", file=sys.stderr)
    with open(OUT, 'a') as fh:
        for i, p in enumerate(todo, 1):
            a = fetch_source(f"{p['id']}v1")
            time.sleep(SLEEP)
            b = fetch_source(f"{p['id']}v{p['v_last']}")
            time.sleep(SLEEP)
            if not a or not b:
                rec = {'id': p['id'], 'arm': p['arm'], 'status': 'source-unavailable'}
            else:
                A, B = clean_numbers(a), clean_numbers(b)
                dropped = sum((A - B).values())
                rec = {
                    'id': p['id'], 'arm': p['arm'], 'v_last': p['v_last'],
                    'comment': p.get('comment', ''), 'status': 'ok',
                    'n_v1': sum(A.values()), 'n_vlast': sum(B.values()),
                    'shared': sum((A & B).values()), 'dropped': dropped,
                    'added': sum((B - A).values()),
                    'dropped_share': round(dropped / max(sum(A.values()), 1), 4),
                }
            fh.write(json.dumps(rec) + '\n')
            fh.flush()
            print(f"[{i}/{len(todo)}] {rec['id']} {rec['status']} "
                  f"n={rec.get('n_v1','-')} drop={rec.get('dropped_share','-')}",
                  file=sys.stderr)
