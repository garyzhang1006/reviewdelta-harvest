#!/usr/bin/env python3
"""Print a labeled-evidence sample of classified drops for hand validation.

Seeded sample of papers with at least 3 drops; for each dropped value the
script prints the classifier's category next to the raw evidence a human
needs to judge it: the v1 context window, whether a numerically equal
variant exists in vN, the vN value at the same anchored cell, and whether
the v1 row label survives. Writes drop_labels_sample.json for the label file.

Usage: python3 drop_labels.py [n_papers]
"""
import json, os, re, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diff_arms as da
from claim_probe import cell_grid, CELL_SPLIT, ROW_SPLIT
from drop_taxonomy import row_labels, ANYNUM
from collections import Counter

n_papers = int(sys.argv[1]) if len(sys.argv) > 1 else 30

pairs = [json.loads(l) for l in open('pairs.jsonl')]
ok = [p for p in pairs if p['status'] == 'ok' and p['n_v1'] > 0 and p['dropped'] >= 3]
rng = random.Random(7)
sample = rng.sample(ok, n_papers)

out = []
for i, p in enumerate(sample, 1):
    v1 = da.fetch_source(f"{p['id']}v1")
    time.sleep(da.SLEEP)
    vn = da.fetch_source(f"{p['id']}v{p['v_last']}")
    time.sleep(da.SLEEP)
    if not v1 or not vn:
        print(f"[{i}/{n_papers}] {p['id']} unavailable", file=sys.stderr)
        continue
    A, B = da.result_numbers(v1), da.result_numbers(vn)
    dropped = list((A - B).elements())
    rng.shuffle(dropped)
    vn_values = {}
    for m in ANYNUM.finditer(vn):
        try:
            vn_values.setdefault(float(m.group(0)), set()).add(m.group(0))
        except ValueError:
            pass
    g1, gn = cell_grid(v1), cell_grid(vn)
    pos1 = {}
    for k, v in g1.items():
        pos1.setdefault(v, []).append(k)
    labs_n = row_labels(vn)
    labs_1 = {}
    for m in da.TABULAR.finditer(v1):
        for row in ROW_SPLIT.split(m.group(0)):
            cells = CELL_SPLIT.split(row)
            if len(cells) >= 2:
                lab = re.sub(r'[\\{}$&%_^~ ]|\d', '', cells[0]).lower()
                if len(lab) >= 4:
                    for n in da.NUM.findall(row):
                        labs_1.setdefault(n, set()).add(lab)
    for val in dropped[:4]:
        fv = float(val)
        variants = sorted(s for s in vn_values.get(fv, ()) if s != val)
        cellhits = [(k, gn[k]) for k in pos1.get(val, ()) if k in gn and gn[k] != val]
        labhit = any(l in labs_n for l in labs_1.get(val, ()))
        if variants:
            cat = 'formatting'
        elif cellhits:
            cat = 'value-change'
        elif labhit:
            cat = 'row-change'
        else:
            cat = 'gone'
        idx = v1.find(val)
        ctx = re.sub(r'\s+', ' ', v1[max(0, idx-70):idx+70])
        out.append({'paper': p['id'], 'value': val, 'classifier': cat,
                    'v1_context': ctx,
                    'vn_variants': variants[:3],
                    'vn_cell_values': [v for _, v in cellhits[:3]],
                    'row_label_survives': labhit})
    print(f"[{i}/{n_papers}] {p['id']} {len(dropped)} drops", file=sys.stderr)

json.dump(out, open('drop_labels_sample.json', 'w'), indent=1)
print(f"# {len(out)} classified drops for hand validation")
for i, o in enumerate(out, 1):
    print(f"{i}\t{o['paper']}\t{o['classifier']}\t{o['value']}\t"
          f"variants={o['vn_variants']}\tcell={o['vn_cell_values']}\t"
          f"row={o['row_label_survives']}\tctx={o['v1_context']}")
