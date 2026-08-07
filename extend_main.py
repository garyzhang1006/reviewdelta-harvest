#!/usr/bin/env python3
"""Stage 12: extend the main corpus for power.

The five 2025 listing months yielded 10,071 unique ids and the original harvest
fetched metadata for 1,396. This fetches abs pages for more of the SAME
candidate pool (no new listing months, so the sampling frame is unchanged),
reassigns arms over the enlarged metadata set, and diffs every new treatment
paper plus a seeded random top-up of new controls. Appends to meta.jsonl and
pairs.jsonl; both are resumable, and report.py recomputes everything.
"""
import json, os, random, sys, time
import harvest as hv
import diff_arms as da

ABS_BUDGET = 700
NEW_CONTROL_DIFFS = 60

if __name__ == '__main__':
    done = {}
    for line in open('meta.jsonl'):
        try:
            r = json.loads(line)
            done[r['id']] = r
        except Exception:
            pass
    cands = []
    for cat in hv.CATS:
        for mo in hv.MONTHS:
            ids = hv.listing_ids(cat, mo)
            cands += ids
            time.sleep(hv.SLEEP)
    uniq = list(dict.fromkeys(cands))
    todo = [i for i in uniq if i not in done]
    random.Random(1).shuffle(todo)
    todo = todo[:ABS_BUDGET]
    print(f"{len(done)} cached, fetching {len(todo)} abs", file=sys.stderr)
    with open('meta.jsonl', 'a') as fh:
        for n, aid in enumerate(todo, 1):
            m = hv.abs_meta(aid)
            if m:
                fh.write(json.dumps(m) + '\n')
                fh.flush()
                done[aid] = m
            if n % 50 == 0:
                print(f"  abs [{n}/{len(todo)}]", file=sys.stderr)
            time.sleep(hv.SLEEP)

    revised = [r for r in done.values() if r['version'] >= 2]
    for r in revised:
        r['arm'] = 'treatment' if hv.VENUE.search(r['comment']) else 'control'
    json.dump({'revised': revised}, open('arms.json', 'w'))
    t = [r for r in revised if r['arm'] == 'treatment']
    c = [r for r in revised if r['arm'] == 'control']
    print(f"arms now {len(t)} treatment / {len(c)} control", file=sys.stderr)

    prev = set()
    for line in open('pairs.jsonl'):
        try:
            prev.add(json.loads(line)['id'])
        except Exception:
            pass
    new_t = [r for r in t if r['id'] not in prev]
    new_c = [r for r in c if r['id'] not in prev]
    random.Random(1).shuffle(new_c)
    plan = new_t + new_c[:NEW_CONTROL_DIFFS]
    print(f"diffing {len(new_t)} new treatment + {min(len(new_c), NEW_CONTROL_DIFFS)} new controls",
          file=sys.stderr)
    with open('pairs.jsonl', 'a') as fh:
        for i, rec in enumerate(plan, 1):
            r = da.diff(rec)
            fh.write(json.dumps(r) + '\n')
            fh.flush()
            print(f"[{i}/{len(plan)}] {r['id']} {r['arm']} {r['status']} "
                  f"drop={r.get('dropped_share', '-')}", file=sys.stderr)
    print("extension done")
