#!/usr/bin/env python3
"""Stage 10: how loose is the verbatim lower bound?

On a stratified 30-paper subsample (10 zero-churn, 10 middle, 10 high under the
document-level metric), refetch v1 and the final version and score churn two
more ways:

1. cell-anchored: inside tabular blocks, key each cell value by (table index,
   row index, column index) and count a v1 cell as surviving only if the same
   value sits at the same key in the final version's table grid. Strict: table
   restructuring counts as churn, so this overstates.
2. collision baseline: for each paper, the share of its v1 numbers that appear
   in the final multiset of a different, size-matched paper. Estimates how much
   of document-level "survival" string coincidence alone would produce.

The truth sits between the document-level metric (understates churn) and the
cell-anchored one (overstates it). Writes claim_probe.json.
"""
import json, sys, time, re
import diff_arms as da

TAB = da.TABULAR
CELL_SPLIT = re.compile(r'(?<!\\)&')
ROW_SPLIT = re.compile(r'\\\\')


def cell_grid(tex):
    """{(table_i, row_i, col_i): value} for every decimal in every tabular."""
    grid = {}
    for ti, m in enumerate(TAB.finditer(tex)):
        body = m.group(0)
        for ri, row in enumerate(ROW_SPLIT.split(body)):
            for ci, cell in enumerate(CELL_SPLIT.split(row)):
                nums = da.NUM.findall(cell)
                if len(nums) == 1:
                    grid[(ti, ri, ci)] = nums[0]
    return grid


if __name__ == '__main__':
    ok = [json.loads(l) for l in open('pairs.jsonl')]
    ok = [r for r in ok if r['status'] == 'ok' and r['n_v1'] > 0]
    zero = [r for r in ok if r['dropped_share'] == 0][:10]
    mid = [r for r in ok if 0 < r['dropped_share'] <= 0.2][:10]
    hi = [r for r in ok if r['dropped_share'] > 0.2][:10]
    sample = zero + mid + hi
    out = []
    for i, p in enumerate(sample, 1):
        v1 = da.fetch_source(f"{p['id']}v1")
        time.sleep(da.SLEEP)
        vn = da.fetch_source(f"{p['id']}v{p['v_last']}")
        time.sleep(da.SLEEP)
        if not v1 or not vn:
            print(f"[{i}/30] {p['id']} unavailable", file=sys.stderr)
            continue
        A, B = da.result_numbers(v1), da.result_numbers(vn)
        g1, gn = cell_grid(v1), cell_grid(vn)
        anchored_total = len(g1)
        anchored_kept = sum(1 for k, v in g1.items() if gn.get(k) == v)
        rec = {
            'id': p['id'], 'doc_share': p['dropped_share'],
            'n_v1': sum(A.values()),
            'v1_numbers': sorted(A.elements()),
            'vn_numbers': sorted(B.elements()),
            'anchored_total': anchored_total,
            'anchored_kept': anchored_kept,
        }
        out.append(rec)
        a_share = 1 - anchored_kept / anchored_total if anchored_total else None
        print(f"[{i}/30] {p['id']} doc={p['dropped_share']:.2f} "
              f"anchored={a_share if a_share is None else round(a_share,2)}",
              file=sys.stderr)
    json.dump(out, open('claim_probe.json', 'w'))
    print(f"wrote claim_probe.json ({len(out)} papers)")
