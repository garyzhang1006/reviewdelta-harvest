#!/usr/bin/env python3
"""Stage 11: out-of-sample replication cells.

The headline corpus is cs.LG+cs.CL, 2025. This script runs the same pipeline
on a cell the corpus never touched and reports the shape (zero / middle / >20%)
and the arm difference there. Usage:

    python3 replicate.py cv25    # cs.CV, March + August 2025
    python3 replicate.py lgcl24  # cs.LG + cs.CL, March + August 2024

Each cell harvests ~250 abs records, labels arms with the same VENUE pattern,
attempts all treatment papers plus a seeded random control sample up to 100
total diffs, and appends to meta_<cell>.jsonl / pairs_<cell>.jsonl. Resumable.
"""
import json, os, random, sys, time
import harvest as hv
import diff_arms as da

CELLS = {
    'cv25': (['cs.CV'], ['2025-03', '2025-08']),
    'lgcl24': (['cs.LG', 'cs.CL'], ['2024-03', '2024-08']),
    'cv25b': (['cs.CV'], ['2025-01', '2025-05']),
}
ABS_BUDGET = 250
DIFF_BUDGET = 100

if __name__ == '__main__':
    cell = sys.argv[1]
    cats, months = CELLS[cell]
    meta_path, pairs_path = f'meta_{cell}.jsonl', f'pairs_{cell}.jsonl'

    done = {}
    if os.path.exists(meta_path):
        for line in open(meta_path):
            try:
                r = json.loads(line)
                done[r['id']] = r
            except Exception:
                pass
    cands = []
    for cat in cats:
        for mo in months:
            ids = hv.listing_ids(cat, mo)
            print(f"listing {cat} {mo}: {len(ids)}", file=sys.stderr)
            cands += ids
            time.sleep(hv.SLEEP)
    uniq = list(dict.fromkeys(cands))
    todo = [i for i in uniq if i not in done]
    random.Random(0).shuffle(todo)
    todo = todo[:max(0, ABS_BUDGET - len(done))]
    print(f"{cell}: {len(uniq)} ids, {len(done)} cached, fetching {len(todo)}",
          file=sys.stderr)
    with open(meta_path, 'a') as fh:
        for n, aid in enumerate(todo, 1):
            m = hv.abs_meta(aid)
            if m:
                fh.write(json.dumps(m) + '\n')
                fh.flush()
                done[aid] = m
            if n % 25 == 0:
                print(f"  abs [{n}/{len(todo)}]", file=sys.stderr)
            time.sleep(hv.SLEEP)

    revised = [r for r in done.values() if r['version'] >= 2]
    for r in revised:
        r['arm'] = 'treatment' if hv.VENUE.search(r['comment']) else 'control'
        r['cat'] = r.get('cat', '')
    t = [r for r in revised if r['arm'] == 'treatment']
    c = [r for r in revised if r['arm'] == 'control']
    print(f"{cell}: {len(done)} meta, {len(revised)} revised, "
          f"{len(t)} treatment / {len(c)} control", file=sys.stderr)

    prev = set()
    if os.path.exists(pairs_path):
        for line in open(pairs_path):
            try:
                prev.add(json.loads(line)['id'])
            except Exception:
                pass
    rng = random.Random(0)
    rng.shuffle(c)
    plan = t + c[:max(0, DIFF_BUDGET - len(t))]
    plan = [r for r in plan if r['id'] not in prev]
    print(f"{cell}: diffing {len(plan)} papers", file=sys.stderr)
    with open(pairs_path, 'a') as fh:
        for i, rec in enumerate(plan, 1):
            r = da.diff(rec)
            fh.write(json.dumps(r) + '\n')
            fh.flush()
            print(f"[{i}/{len(plan)}] {r['id']} {r['arm']} {r['status']} "
                  f"drop={r.get('dropped_share', '-')}", file=sys.stderr)
    print(f"{cell} done")
