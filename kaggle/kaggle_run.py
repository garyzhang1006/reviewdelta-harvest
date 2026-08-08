#!/usr/bin/env python3
"""
Kaggle session driver for the ReviewDelta corpus expansion.

Kaggle gives a CPU session of at most ~12 hours with internet access, one
shared egress IP, and outputs that persist only from /kaggle/working. This
script fits the pipeline to those constraints:

  - clones the public repo, then overlays any previous run's output (attach
    the prior kernel output as an input dataset) so every session resumes
    instead of restarting;
  - harvests metadata with --cap 1000 per category-month, reproducing the
    original 10,071-candidate listing frame rather than the ~2.5x larger
    uncapped one, so the source fetch fits a single session;
  - fetches as ONE polite stream (3s delay). Kaggle traffic shares an IP
    range with every other kernel, so this host gets no sharding;
  - stops fetching at a hard deadline with time reserved for the probes and
    the report, then copies everything worth keeping to /kaggle/working.

Child output is re-emitted line by line: Kaggle drops long subprocess stdout,
so an empty log on a RUNNING kernel would otherwise look like a hang.
"""
import os
import shutil
import subprocess
import sys
import time

REPO = "https://github.com/garyzhang1006/reviewdelta-harvest.git"
CONTACT = "jakobingerbrigsten@gmail.com"   # arXiv etiquette: be reachable
BUDGET_H = float(os.environ.get("KAGGLE_BUDGET_H", "11.0"))
FETCH_RESERVE_H = 2.5      # kept back for probes + report + output copy
WORK = "/kaggle/working"
T0 = time.time()

# Files that carry run state forward between sessions.
STATE_FILES = [
    "meta.jsonl", "arms.json", "pairs.jsonl",
]
STATE_GLOBS = ["pairs_shard", ".source_cache"]


def left_h():
    return BUDGET_H - (time.time() - T0) / 3600


def run(cmd, env_extra=None, cwd=None):
    """Stream a child's output through our stdout so Kaggle keeps the log."""
    print(f"\n$ {' '.join(cmd)}   [{left_h():.1f}h left]", flush=True)
    env = dict(os.environ)
    env["REVIEWDELTA_CONTACT"] = CONTACT
    if env_extra:
        env.update(env_extra)
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert p.stdout is not None
    for line in p.stdout:
        print(line, end="", flush=True)
    p.wait()
    print(f"[exit {p.returncode}]", flush=True)
    return p.returncode


def save_outputs(src="."):
    """Copy resumable state and results into /kaggle/working."""
    for name in os.listdir(src):
        keep = (name in STATE_FILES
                or any(name.startswith(g) or name == g for g in STATE_GLOBS)
                or name == "results")
        if not keep:
            continue
        s, d = os.path.join(src, name), os.path.join(WORK, name)
        try:
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        except Exception as e:
            print(f"  save {name}: {e}", flush=True)
    print("outputs saved to /kaggle/working", flush=True)


def main():
    os.chdir("/kaggle" if os.path.isdir("/kaggle") else ".")
    if os.path.isdir("repo"):
        shutil.rmtree("repo")
    if run(["git", "clone", "--depth", "1", REPO, "repo"]) != 0:
        sys.exit("clone failed; no internet?")
    os.chdir("repo")

    # Resume: overlay prior state from any attached input dataset. Attached
    # outputs land one level down, /kaggle/input/<slug>/<file>.
    inp = "/kaggle/input"
    slugs = [os.path.join(inp, d) for d in os.listdir(inp)] if os.path.isdir(inp) else []
    for slug in slugs:
        if not os.path.isdir(slug):
            continue
        for name in os.listdir(slug):
            if not (name in STATE_FILES
                    or any(name.startswith(g) for g in STATE_GLOBS)):
                continue
            s = os.path.join(slug, name)
            try:
                if os.path.isdir(s):
                    shutil.copytree(s, name, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, name)
                print(f"resumed {name} from {slug}", flush=True)
            except Exception as e:
                print(f"  resume {name}: {e}", flush=True)

    try:
        # Metadata. The validate gate decides API vs the slow proven scraper;
        # the scraper does not fit this session alongside the fetch, so a
        # validation failure ends the run with a clear message instead.
        if not (os.path.exists("arms.json") and os.path.getsize("arms.json") > 100000):
            if run([sys.executable, "arxiv_meta.py", "--validate"]) != 0:
                save_outputs()
                sys.exit("API validation failed. Run harvest.py on a host "
                         "without a session cap; Kaggle cannot fit it.")
            if run([sys.executable, "arxiv_meta.py", "--cap", "1000"]) != 0:
                save_outputs()
                sys.exit("metadata harvest failed")
        else:
            print("arms.json present; metadata stage skipped", flush=True)

        # Source fetch: single polite stream, hard deadline.
        fetch_s = max(0, (left_h() - FETCH_RESERVE_H) * 3600)
        print(f"fetch budget: {fetch_s / 3600:.1f}h", flush=True)
        run([sys.executable, "fetch_pairs.py", "--shard", "0", "--nshards", "1",
             "--sleep", "3.0", "--max-seconds", str(int(fetch_s))])

        run([sys.executable, "fetch_pairs.py", "--merge"])

        # Probes + report only if the fetch actually finished the corpus;
        # on a partial fetch the next session resumes first.
        if left_h() > 1.5:
            run([sys.executable, "run_all.py", "--stages", "probes", "report"],
                env_extra={"REVIEWDELTA_SLEEP": "1.0"})
        else:
            print("out of time for probes; rerun this kernel with this "
                  "output attached to resume", flush=True)
    finally:
        save_outputs()


if __name__ == "__main__":
    main()
