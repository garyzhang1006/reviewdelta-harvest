#!/usr/bin/env python3
"""
One command that takes a clone to a full set of results.

Runs five stages in order, skipping any whose output already exists, so a killed
job resumes by rerunning the same command:

  1. meta    arxiv_meta.py, with automatic fallback to harvest.py if the API
             validation gate fails
  2. fetch   sharded source fetch, populating pairs.jsonl and the source cache
  3. merge   fold shard files into pairs.jsonl
  4. probes  the seven robustness analyses, reading the cache instead of
             refetching
  5. report  report.py, printing every number the paper quotes

The source cache is what makes stage 4 cheap. Every probe calls
diff_arms.fetch_source on papers stage 2 already fetched, so without the cache
each probe repeats the whole network pass. With it, stage 4 is local disk.

Usage:
  export REVIEWDELTA_CONTACT='you@example.edu'
  python3 run_all.py                      # everything, resuming as needed
  python3 run_all.py --plan                # print what would run, touch nothing
  python3 run_all.py --stages meta fetch   # a subset
  python3 run_all.py --force probes        # redo a completed stage
  python3 run_all.py --nshards 8           # parallel fetch (default 8)

Nothing here launches jobs on a scheduler. To use SLURM, run stage 2 through
submit_fetch.sbatch and then `python3 run_all.py --stages merge probes report`.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

STATE = ".run_all_state.json"

# stage -> (script, output file that proves it finished)
PROBES = [
    ("claim_probe.py", "claim_probe.json"),
    ("anchored_full.py", "anchored_full.jsonl"),
    ("recall_probe.py", "recall_probe.jsonl"),
    ("drop_taxonomy.py", "drop_taxonomy.jsonl"),
    ("baseline.py", "baseline.jsonl"),
    ("longitudinal.py", "longitudinal.jsonl"),
    ("fp_survival.py", "fp_survival.json"),
]

ALL_STAGES = ["meta", "fetch", "merge", "probes", "report"]


def load_state():
    if os.path.exists(STATE):
        try:
            return json.load(open(STATE))
        except Exception:
            pass
    return {"done": [], "log": []}


def save_state(st):
    json.dump(st, open(STATE, "w"), indent=1)


def say(msg):
    print(f"[run_all {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, env=None):
    """Run a child and stream its output. Returns the exit code."""
    say(f"$ {' '.join(cmd)}")
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run(cmd, env=e)
    return p.returncode


def preflight(a):
    """Fail loudly and early rather than three hours into a fetch."""
    problems = []

    if not a.plan:
        contact = os.environ.get("REVIEWDELTA_CONTACT", "")
        if "@" not in contact:
            problems.append(
                "REVIEWDELTA_CONTACT is unset or has no '@'.\n"
                "    arXiv blocks anonymous bulk fetchers.\n"
                "    export REVIEWDELTA_CONTACT='you@example.edu'")

    if sys.version_info < (3, 8):
        problems.append(f"Python {sys.version_info.major}.{sys.version_info.minor} "
                        "is too old; need 3.8+")

    for mod in ("numpy", "scipy"):
        try:
            __import__(mod)
        except ImportError:
            problems.append(f"missing module {mod!r} (report.py needs it): "
                            f"pip install {mod}")

    for f in ("arms.json", "diff_arms.py", "report.py"):
        if not os.path.exists(f):
            problems.append(f"missing {f}: run from inside the repo directory")

    # The cache holds the .tex of every version fetched. Roughly 60 KB gzipped
    # per version, two versions per paper, so ~0.5 GB for a 4,000-paper corpus.
    try:
        free_gb = shutil.disk_usage(".").free / 1e9
        if free_gb < 3:
            problems.append(f"only {free_gb:.1f} GB free; the source cache wants ~1 GB "
                            "and arXiv tarballs stream through /tmp")
    except Exception:
        pass

    if problems:
        print("\nPREFLIGHT FAILED\n", file=sys.stderr)
        for p in problems:
            print("  - " + p, file=sys.stderr)
        print(file=sys.stderr)
        sys.exit(1)
    say("preflight ok")


def n_usable(path="pairs.jsonl"):
    if not os.path.exists(path):
        return 0
    n = 0
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("status") == "ok" and r.get("n_v1", 0) > 0:
            n += 1
    return n


def stage_meta(a):
    """API metadata, falling back to the proven scraper if validation fails."""
    if run([sys.executable, "arxiv_meta.py", "--validate"]) != 0:
        say("API validation FAILED; falling back to harvest.py (slower, proven)")
        return run([sys.executable, "harvest.py", str(a.meta_budget)]) == 0
    args = [sys.executable, "arxiv_meta.py"]
    args += ["--all-2025"] if a.all_2025 else []
    return run(args) == 0


def stage_fetch(a):
    if a.nshards <= 1:
        return run([sys.executable, "fetch_pairs.py",
                    "--shard", "0", "--nshards", "1"]) == 0

    say(f"launching {a.nshards} shards in parallel")
    os.makedirs("logs", exist_ok=True)
    procs = []
    for i in range(a.nshards):
        log = open(f"logs/shard{i:02d}.log", "a")
        procs.append((i, subprocess.Popen(
            [sys.executable, "fetch_pairs.py",
             "--shard", str(i), "--nshards", str(a.nshards)],
            stdout=log, stderr=subprocess.STDOUT)))
    say("shards running; progress in logs/shard*.log")
    bad = []
    for i, p in procs:
        if p.wait() != 0:
            bad.append(i)
    if bad:
        say(f"shards {bad} exited nonzero; rerun to resume (finished papers are kept)")
        return False
    return True


def stage_merge(a):
    return run([sys.executable, "fetch_pairs.py", "--merge"]) == 0


def stage_probes(a):
    """The seven robustness analyses, served from the warm source cache.

    Sleep drops to 1s. Cache hits return instantly, so the sleep is the cost;
    a probe paper missing from the cache (the held-out cells, which the main
    fetch never touches) falls back to a real arXiv fetch, and 1s is the
    fastest polite rate for those misses.
    """
    env = {"REVIEWDELTA_SLEEP": "1.0"}
    failed = []
    for script, out in PROBES:
        if not os.path.exists(script):
            say(f"skip {script}: not present")
            continue
        before = os.path.getsize(out) if os.path.exists(out) else -1
        if run([sys.executable, script], env=env) != 0:
            failed.append(script)
            say(f"{script} FAILED; continuing with the rest")
            continue
        after = os.path.getsize(out) if os.path.exists(out) else -1
        say(f"{script} -> {out} ({before} -> {after} bytes)")
    if failed:
        say(f"probes with failures: {failed}. Rerun with --force probes after fixing.")
    return not failed


def stage_report(a):
    say("writing results/report.txt")
    os.makedirs("results", exist_ok=True)
    with open("results/report.txt", "w") as fh:
        p = subprocess.run([sys.executable, "report.py"],
                           stdout=fh, stderr=subprocess.STDOUT)
    if p.returncode == 0:
        print(open("results/report.txt").read())
    return p.returncode == 0


STAGE_FN = {
    "meta": stage_meta,
    "fetch": stage_fetch,
    "merge": stage_merge,
    "probes": stage_probes,
    "report": stage_report,
}


def stage_is_done(name):
    """Cheap completion checks, so a rerun skips finished work."""
    if name == "meta":
        return os.path.exists("arms.json") and os.path.getsize("arms.json") > 1000
    if name == "fetch":
        return bool(glob.glob("pairs_shard*.jsonl")) or n_usable() > 0
    if name == "merge":
        return n_usable() > 0 and not glob.glob("pairs_shard*.jsonl")
    if name == "probes":
        return all(os.path.exists(o) for _, o in PROBES)
    if name == "report":
        return os.path.exists("results/report.txt")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=ALL_STAGES, choices=ALL_STAGES)
    ap.add_argument("--force", nargs="*", default=[], choices=ALL_STAGES,
                    help="rerun these even if their output exists")
    ap.add_argument("--nshards", type=int, default=8)
    ap.add_argument("--all-2025", action="store_true", default=True,
                    help="harvest every month of 2025 (default)")
    ap.add_argument("--meta-budget", type=int, default=8000,
                    help="abs pages to fetch if the API path fails")
    ap.add_argument("--plan", action="store_true",
                    help="print the plan and exit without running anything")
    a = ap.parse_args()

    preflight(a)
    st = load_state()

    plan = []
    for s in a.stages:
        done = stage_is_done(s) and s not in a.force
        plan.append((s, "SKIP (done)" if done else "RUN"))

    say("plan:")
    for s, what in plan:
        print(f"    {s:8s} {what}")
    if a.plan:
        say("--plan given, nothing executed")
        return

    say(f"corpus before: {n_usable()} usable pairs")
    t0 = time.time()
    for s, what in plan:
        if what.startswith("SKIP"):
            continue
        say(f"=== stage {s} ===")
        ok = STAGE_FN[s](a)
        if not ok:
            say(f"stage {s} did not complete. Fix, then rerun the same command; "
                "finished work is kept.")
            save_state(st)
            sys.exit(1)
        if s not in st["done"]:
            st["done"].append(s)
        st["log"].append({"stage": s, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
        save_state(st)

    say(f"corpus after: {n_usable()} usable pairs")
    say(f"all requested stages finished in {(time.time() - t0) / 3600:.2f} h")
    say("results/report.txt holds every number")


if __name__ == "__main__":
    main()
