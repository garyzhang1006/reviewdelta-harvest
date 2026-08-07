#!/usr/bin/env python3
"""
Stage 2b: fetch v1 and latest LaTeX source for every revised paper, sharded.

diff_arms.py does the same job in one serial process. At two source fetches per
paper and a 4-second delay, a 4,000-paper corpus takes about 9 hours, which is
longer than most cluster job limits. This module splits the work into N shards
that run as independent array tasks and merge afterwards.

The extraction itself is imported from diff_arms, never reimplemented, so shard
output is byte-compatible with the existing pairs.jsonl. If you change how a
number is recognised, change it in diff_arms.py and both paths follow.

Sharding is deterministic: papers are sorted by id and dealt round-robin, so
shard k always gets the same papers no matter how many times you rerun or in
what order tasks start. Each shard appends to its own file and skips ids it has
already finished, so a killed task resumes instead of restarting.

Usage:
  python3 fetch_pairs.py --shard 0 --nshards 8          # one array task
  python3 fetch_pairs.py --merge                        # after all shards finish
  python3 fetch_pairs.py --shard 0 --nshards 8 --limit 5   # tiny trial run

Set REVIEWDELTA_CONTACT to an email before running. arXiv blocks anonymous bulk
fetchers and warns contactable ones; this script refuses to start without it.
"""
import argparse
import glob
import json
import os
import sys
import time

import diff_arms as da


def require_contact():
    """arXiv's bulk-access etiquette: identify yourself or do not run."""
    email = os.environ.get("REVIEWDELTA_CONTACT", "").strip()
    if not email or "@" not in email:
        sys.exit(
            "REVIEWDELTA_CONTACT is unset.\n"
            "arXiv asks bulk fetchers to be reachable, and blocks anonymous ones.\n"
            "  export REVIEWDELTA_CONTACT='you@example.edu'\n"
        )
    ua = f"ReviewDelta/1.0 (academic research; {email})"
    da.UA = {"User-Agent": ua}
    # harvest.py holds its own copy of the header for the metadata stage.
    try:
        import harvest as hv
        hv.UA = {"User-Agent": ua}
    except Exception:
        pass
    return ua


def plan(nshards, shard, arms_path="arms.json", done_paths=("pairs.jsonl",)):
    """The papers this shard owns and has not already finished."""
    revised = json.load(open(arms_path))["revised"]

    done = set()
    for pat in done_paths:
        for path in glob.glob(pat):
            if not os.path.exists(path):
                continue
            for line in open(path):
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass

    # Sort before dealing so the split does not depend on dict ordering.
    pool = sorted((r for r in revised if r["id"] not in done),
                  key=lambda r: r["id"])
    return [r for i, r in enumerate(pool) if i % nshards == shard], len(done)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N papers (trial runs only)")
    ap.add_argument("--merge", action="store_true",
                    help="fold every shard file into pairs.jsonl and exit")
    ap.add_argument("--sleep", type=float, default=da.SLEEP,
                    help="seconds between source fetches (default 4, arXiv-friendly)")
    a = ap.parse_args()

    if a.merge:
        merge()
        return

    if not (0 <= a.shard < a.nshards):
        sys.exit(f"--shard must be in [0, {a.nshards})")

    ua = require_contact()
    da.SLEEP = a.sleep

    out = f"pairs_shard{a.shard:02d}.jsonl"
    todo, ndone = plan(a.nshards, a.shard,
                       done_paths=("pairs.jsonl", "pairs_shard*.jsonl"))
    if a.limit:
        todo = todo[:a.limit]

    print(f"UA: {ua}", file=sys.stderr)
    print(f"shard {a.shard}/{a.nshards}: {len(todo)} papers to fetch "
          f"({ndone} already done corpus-wide)", file=sys.stderr)
    print(f"estimated wall clock: {len(todo) * 2 * a.sleep / 3600:.1f} h",
          file=sys.stderr)

    t0 = time.time()
    with open(out, "a") as fh:
        for i, rec in enumerate(todo, 1):
            r = da.diff(rec)
            fh.write(json.dumps(r) + "\n")
            fh.flush()          # a killed task loses at most the current paper
            if i % 10 == 0 or i == len(todo):
                rate = (time.time() - t0) / i
                print(f"  [{i}/{len(todo)}] {r['id']} {r['arm']} {r['status']} "
                      f"drop={r.get('dropped_share', '-')} "
                      f"eta={(len(todo) - i) * rate / 3600:.1f}h",
                      file=sys.stderr, flush=True)
    print(f"shard {a.shard} done -> {out}", file=sys.stderr)


def merge(out="pairs.jsonl"):
    """Fold shard files into pairs.jsonl, keeping one record per id."""
    seen, rows = set(), []
    if os.path.exists(out):
        for line in open(out):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r["id"] not in seen:
                seen.add(r["id"])
                rows.append(r)
    before = len(rows)

    shards = sorted(glob.glob("pairs_shard*.jsonl"))
    for path in shards:
        for line in open(path):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r["id"] not in seen:
                seen.add(r["id"])
                rows.append(r)

    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    ok = [r for r in rows if r.get("status") == "ok" and r.get("n_v1", 0) > 0]
    t = sum(1 for r in ok if r["arm"] == "treatment")
    print(f"merged {len(shards)} shard files: {before} -> {len(rows)} records")
    print(f"usable pairs: {len(ok)}  (treatment {t}, control {len(ok) - t})")
    print("now run: python3 report.py")


if __name__ == "__main__":
    main()
