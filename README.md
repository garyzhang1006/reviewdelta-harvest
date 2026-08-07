# ReviewDelta harvest

Pipeline for tracking every reported number across arXiv version pairs, plus the
cluster scripts for expanding the corpus.

The existing corpus holds 311 usable pairs. That is enough to describe the shape
of revision but not enough to bound the arm comparison: the minimum detectable
effect sits at 0.097 against a mean per-paper churn of 0.182, so only effects
above roughly half the mean are excluded. Expanding the corpus is the point of
this repo.

## What expansion buys

| corpus | treatment | control | MDE (80% power) | as share of 0.182 mean |
|---|---|---|---|---|
| current | 136 | 175 | 0.097 | 53% |
| + unfetched controls only | 136 | 596 | 0.081 | 44% |
| full candidate frame | ~617 | ~2900 | 0.038 | 21% |

The middle row is the trap. Controls are cheap and nearly useless here, because
the 136-paper treatment arm is what binds. More treatment papers only come from
harvesting more of the candidate frame, which is what `arxiv_meta.py` makes
affordable.

## Quickstart

Get it onto the machine:

```bash
git clone https://github.com/garyzhang1006/reviewdelta-harvest.git
cd reviewdelta-harvest
```

Install the two dependencies (the fetch stages use only the standard library;
`report.py` needs these):

```bash
pip install --user numpy scipy
```

Set your contact address and run:

```bash
export REVIEWDELTA_CONTACT='you@example.edu'   # required, see "arXiv etiquette"
python3 run_all.py
```

Expect 5 to 7 hours, almost all of it waiting on arXiv's 3-second courtesy
delay rather than computing. Run it under `tmux` or `nohup` so a dropped ssh
session does not kill it:

```bash
tmux new -s rd 'python3 run_all.py 2>&1 | tee run.log'
```

Watch it from another shell:

```bash
tail -f logs/shard*.log            # per-shard fetch progress and ETA
cat pairs_shard*.jsonl | wc -l     # papers finished so far
```

When it finishes, every number lands in `results/report.txt`:

```bash
less results/report.txt
```

That is the whole thing. `run_all.py` runs five stages in order (metadata,
sharded fetch, merge, the seven robustness probes, report), skips any stage
whose output already exists, and writes `results/report.txt`. Kill it and rerun
the same command to resume; finished papers are never refetched.

```bash
python3 run_all.py --plan               # show what would run, touch nothing
python3 run_all.py --stages merge report
python3 run_all.py --force probes       # redo a finished stage
python3 run_all.py --nshards 16         # wider fetch parallelism
```

On SLURM, run the fetch as an array job and let `run_all.py` do the rest:

```bash
mkdir -p logs && sbatch --export=ALL submit_fetch.sbatch
python3 run_all.py --stages merge probes report
```

`report.py` alone reproduces the current paper's numbers from the committed data
with no network access.

## Why the probes are cheap

All seven robustness probes call `diff_arms.fetch_source` on papers the main
fetch already retrieved. Without a cache, each probe repeats the entire network
pass, so a full rerun costs eight passes rather than one.

`diff_arms.py` therefore caches the post-comment-strip `.tex` per version under
`.source_cache/`, gzipped and sharded by id prefix. A cache hit returns a string
byte-identical to what the fetch would have returned, so extraction is
unaffected. Writes go through a temp file and `os.replace`, so parallel shards
never read a half-written entry, and a truncated entry is deleted and refetched
rather than trusted.

`run_all.py` drops `REVIEWDELTA_SLEEP` to 0.2s for the probe stage, since those
read local disk. Set `REVIEWDELTA_CACHE=''` to disable caching entirely.

## Read this before running the metadata stage

`arxiv_meta.py` is **unverified as shipped**. Nobody has run it against the live
API. It replaces `harvest.py`'s per-paper `/abs/` scraping with batched API
calls, which takes the metadata stage from about 10,000 requests to about six.

`harvest.py`'s own docstring says the API "rate-limits to the point of being
unusable". That is true of per-id querying, and this module never does that. It
pages a category-plus-date search and reads slices of up to 2,000 entries. The
claim is still untested, so `--validate` exists: it fetches one small slice,
asserts that versions, comments and categories all parse, then cross-checks three
ids against `harvest.abs_meta()`. It exits nonzero on any mismatch.

If validation fails, fall back to the proven path:

```bash
python3 harvest.py 8000        # slow but known to work, ~8h at 3s per request
```

Everything downstream is unaffected. Both paths write the same `meta.jsonl`
records.

## Sharding and resume

`fetch_pairs.py` deals papers round-robin after sorting by id, so shard *k* owns
the same papers on every rerun regardless of task start order. Each shard appends
to its own `pairs_shard##.jsonl` and flushes after each paper, so a killed task
loses at most one paper and resumes where it stopped. Rerunning a finished shard
is a no-op.

Eight shards means eight concurrent arXiv clients at one request per four seconds
each. If your cluster routes all outbound traffic through a single NAT address,
use four instead. Losing the corpus to a throttle costs more than the extra hours.

Merging is idempotent and keeps the first record for any duplicated id.

## arXiv etiquette

`fetch_pairs.py` refuses to start unless `REVIEWDELTA_CONTACT` holds an email.
arXiv asks bulk fetchers to be reachable, warns the ones that are, and blocks the
ones that are not. The address ends up in the User-Agent on every request.

Source files come from `arxiv.org/e-print/`, roughly 4 MB per paper, so a
4,000-paper run moves about 30 GB. For much larger pulls, use arXiv's S3 bulk
access instead; it is requester-pays and ships 2.9 TB in 500 MB tars, which only
makes sense when you want most of the archive rather than a filtered slice.

## Files

**Harvest and diff**
- `arxiv_meta.py`: batched API metadata (new, validate first)
- `harvest.py`: original `/abs/` scraper, the proven fallback
- `diff_arms.py`: source fetch, number extraction, per-paper diff. Extraction
  lives here and nowhere else; both fetch paths import it
- `fetch_pairs.py`: sharded resumable driver, plus `--merge`
- `extend_main.py`: the earlier single-process extension

**Cluster**
- `submit_fetch.sbatch`: SLURM array job, 8 tasks, 1 CPU and 2 GB each
- `run_local.sh`: same sharding with plain background processes

**Analysis** (all offline)
- `report.py`: prints every quoted number from the committed data
- `anchored_full.py`, `baseline.py`, `recall_probe.py`, `drop_taxonomy.py`,
  `claim_probe.py`, `fp_survival.py`: the robustness probes
- `precision_labels.py`, `drop_labels.py`, `sample_contexts.py`: label handling
- `validate_arms.py`, `replicate.py`, `refetch.py`, `rerun_clean.py`

**Data**
- `meta.jsonl`, `arms.json`: metadata and arm assignment
- `pairs.jsonl`, `pairs_clean.jsonl`: per-paper diffs, both extraction rules
- `pairs_cv25.jsonl`, `pairs_lgcl24.jsonl`, `pairs_cv25b.jsonl`: held-out cells
- `precision*.json`, `drop_labels*.json`: hand labels and blind second passes
- `prereg_cv25b.md`: the prediction recorded before the second cs.CV fetch

## Requirements

Python 3.8 or newer, `numpy` and `scipy` for `report.py`, `matplotlib` for
`make_figure.py`. The fetch stages use only the standard library.

```bash
pip install --user numpy scipy matplotlib
```

`run_all.py` checks all of this in a preflight pass and refuses to start with a
list of what is missing, rather than failing three hours into a fetch.

## If something goes wrong

| symptom | cause | fix |
|---|---|---|
| `PREFLIGHT FAILED ... REVIEWDELTA_CONTACT` | contact address unset | `export REVIEWDELTA_CONTACT='you@example.edu'` |
| `VALIDATION FAILED` from `arxiv_meta.py` | API path is wrong for this host | nothing; `run_all.py` falls back to `harvest.py` on its own |
| shards exit nonzero | arXiv throttled, or the job hit a walltime limit | rerun the same command; finished papers are kept |
| `429` or `403` in a shard log | too many concurrent clients | rerun with `--nshards 4` |
| a probe fails | usually a missing input from an earlier stage | `python3 run_all.py --force probes` after fixing |
| want to start clean | stale state | `rm -rf .run_all_state.json results/` (keep `.source_cache/`) |

Deleting `.source_cache/` is safe but expensive: it forces every probe to
refetch from arXiv.
