#!/usr/bin/env python3
"""
Stage 4: sample extracted numbers with their surrounding LaTeX so a human can
judge whether each is a reported result.

Precision here is the number the corpus lives or dies on. A pipeline that
counts hyperparameters and axis ticks as results would show churn that has
nothing to do with what a paper claims. We sample uniformly from the numbers
the extractor actually accepted, print 120 characters of context per number,
and label them by hand.

Usage: python3 sample_contexts.py [n] > contexts.txt
"""
import json, random, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diff_arms import fetch_source, NUM, TABULAR, PCT_PM

n_want = int(sys.argv[1]) if len(sys.argv) > 1 else 200
# optional second arg: 'tranche2' restricts to papers beyond the first 272
# attempts and writes precision_sample2.json instead.
tranche2 = len(sys.argv) > 2 and sys.argv[2] == 'tranche2'

pairs = [json.loads(l) for l in open('pairs.jsonl')]
if tranche2:
    first = {json.loads(l)['id'] for l in open('pairs.jsonl').readlines()[:272]}
    pairs = [p for p in pairs if p['id'] not in first]
ok = [p for p in pairs if p['status'] == 'ok' and p['n_v1'] > 0]
rng = random.Random(2 if tranche2 else 1)
rng.shuffle(ok)

out, papers_used = [], 0
for p in ok:
    if len(out) >= n_want:
        break
    tex = fetch_source(f"{p['id']}v1")
    if not tex:
        continue
    papers_used += 1
    hits, spans = [], []
    for m in TABULAR.finditer(tex):
        blk, s0 = m.group(0), m.start()
        spans.append(m.span())
        for mm in NUM.finditer(blk):
            hits.append(('tabular', s0 + mm.start(), mm.group(1)))
    for mm in PCT_PM.finditer(tex):
        # mirror result_numbers: in-table percentages are already tabular hits
        if not any(a <= mm.start() < b for a, b in spans):
            hits.append(('pct_pm', mm.start(), mm.group(1)))
    if not hits:
        continue
    rng.shuffle(hits)
    for kind, pos, val in hits[:4]:
        ctx = re.sub(r'\s+', ' ', tex[max(0, pos - 60):pos + 60])
        out.append({'paper': p['id'], 'kind': kind, 'value': val, 'context': ctx})

print(f"# {len(out)} sampled numbers from {papers_used} papers")
print("# label each: R = reported result, N = not a result\n")
for i, o in enumerate(out[:n_want], 1):
    print(f"{i}\t{o['paper']}\t{o['kind']}\t{o['value']}\t{o['context']}")
json.dump(out[:n_want], open('precision_sample2.json' if tranche2 else 'precision_sample.json', 'w'), indent=1)
