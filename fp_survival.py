#!/usr/bin/env python3
"""
Do the labeled false positives actually survive revision?

Section 3 argues false positives (hyperparameters, model-name digits) are
version-stable, so they pad the churn denominator and pull estimates down.
That was an inference from their categories. This measures it: for each of
the labeled false positives in both precision tranches, fetch the paper's
final version and check whether the exact value still appears.

Writes fp_survival.json. Usage: python3 fp_survival.py
"""
import json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diff_arms import fetch_source, SLEEP

fps = []
s1 = json.load(open('precision_sample.json'))
p1 = json.load(open('precision.json'))
for cause, idxs in p1['false_positives'].items():
    for i in idxs:
        rec = s1[i - 1]
        fps.append({'tranche': 1, 'index': i, 'cause': cause,
                    'paper': rec['paper'], 'value': rec['value']})
s2 = json.load(open('precision_sample2.json'))
p2 = json.load(open('precision2.json'))
for cause, idxs in p2['not_result'].items():
    for i in idxs:
        rec = s2[i - 1]
        fps.append({'tranche': 2, 'index': i, 'cause': cause,
                    'paper': rec['paper'], 'value': rec['value']})

vlast = {p['id']: p['v_last'] for p in map(json.loads, open('pairs.jsonl'))
         if p.get('v_last')}
papers = sorted({f['paper'] for f in fps})
print(f"{len(fps)} false positives across {len(papers)} papers", file=sys.stderr)

cache = {}
for i, pid in enumerate(papers, 1):
    cache[pid] = fetch_source(f"{pid}v{vlast[pid]}")
    time.sleep(SLEEP)
    print(f"[{i}/{len(papers)}] {pid} {'ok' if cache[pid] else 'FAIL'}",
          file=sys.stderr)

for f in fps:
    tex = cache[f['paper']]
    if not tex:
        f['survives'] = None
        continue
    f['survives'] = bool(
        re.search(r'(?<![\w.])' + re.escape(f['value']) + r'(?![\w])', tex))

known = [f for f in fps if f['survives'] is not None]
rate = sum(f['survives'] for f in known) / len(known)
out = {'n': len(fps), 'checked': len(known), 'survival_rate': round(rate, 4),
       'by_cause': {}, 'items': fps}
for f in known:
    d = out['by_cause'].setdefault(f['cause'], [0, 0])
    d[0] += f['survives']
    d[1] += 1
json.dump(out, open('fp_survival.json', 'w'), indent=1)
print(f"false-positive survival: {sum(f['survives'] for f in known)}/{len(known)} "
      f"= {rate:.1%}")
for c, (s, n) in sorted(out['by_cause'].items(), key=lambda kv: -kv[1][1]):
    print(f"  {c}: {s}/{n}")
