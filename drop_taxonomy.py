#!/usr/bin/env python3
"""What is a dropped number, actually?

The estimand merges four events the verbatim matcher cannot separate: a value
changed for the same claim, a number deleted with its claim, a table
restructured so the value moved, and a formatting-only rewrite (91.2 becoming
91.20). This pass classifies every dropped v1 number mechanically:

  formatting   a numerically equal string (trailing zeros, added precision)
               exists in the final version
  value-change the number's (table, row, column) cell survives with a
               different value at the same key
  row-change   the cell is gone but the row's text label survives in some
               final-version table row, which now carries different numbers
  gone         none of the above: the row label is not found either

Also prints a seeded 150-drop sample with context for hand labeling.
Resumable: appends to drop_taxonomy.jsonl.
Usage: python3 drop_taxonomy.py
"""
import json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diff_arms as da
from claim_probe import cell_grid, CELL_SPLIT, ROW_SPLIT
from collections import Counter

OUT = 'drop_taxonomy.jsonl'
ANYNUM = re.compile(r'\d+\.?\d*')


def row_labels(tex):
    """Text label (first cell, stripped of math/commands) -> present flag."""
    labels = set()
    for m in da.TABULAR.finditer(tex):
        for row in ROW_SPLIT.split(m.group(0)):
            cells = CELL_SPLIT.split(row)
            if len(cells) >= 2:
                lab = re.sub(r'[\\{}$&%_^~ ]|\d', '', cells[0]).lower()
                if len(lab) >= 4:
                    labels.add(lab)
    return labels


def classify(paper):
    v1 = da.fetch_source(f"{paper['id']}v1")
    time.sleep(da.SLEEP)
    vn = da.fetch_source(f"{paper['id']}v{paper['v_last']}")
    time.sleep(da.SLEEP)
    if not v1 or not vn:
        return {'id': paper['id'], 'status': 'source-unavailable'}
    A, B = da.result_numbers(v1), da.result_numbers(vn)
    dropped = list((A - B).elements())
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
    counts = Counter()
    ctx = []
    for val in dropped:
        try:
            fv = float(val)
        except ValueError:
            fv = None
        if fv is not None and any(s != val for s in vn_values.get(fv, ())):
            cat = 'formatting'
        elif any(k in gn and gn[k] != val for k in pos1.get(val, ())):
            cat = 'value-change'
        elif any(l in labs_n for l in labs_1.get(val, ())):
            cat = 'row-change'
        else:
            cat = 'gone'
        counts[cat] += 1
        ctx.append((cat, val))
    return {'id': paper['id'], 'status': 'ok', 'n_dropped': len(dropped),
            'counts': dict(counts),
            'sample': [{'cat': c, 'value': v} for c, v in ctx[:40]]}


if __name__ == '__main__':
    pairs = [json.loads(l) for l in open('pairs.jsonl')]
    ok = [p for p in pairs if p['status'] == 'ok' and p['n_v1'] > 0
          and p['dropped'] > 0]
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                done.add(json.loads(line)['id'])
            except Exception:
                pass
    todo = [p for p in ok if p['id'] not in done]
    print(f"{len(done)} done; {len(todo)} papers with drops to fetch",
          file=sys.stderr)
    with open(OUT, 'a') as fh:
        for i, p in enumerate(todo, 1):
            r = classify(p)
            fh.write(json.dumps(r) + '\n')
            fh.flush()
            print(f"[{i}/{len(todo)}] {p['id']} {r['status']} "
                  f"{r.get('counts', '')}", file=sys.stderr)
