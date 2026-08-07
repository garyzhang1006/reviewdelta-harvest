#!/usr/bin/env python3
"""
Stage 2: fetch v1 and latest LaTeX source for sampled papers in both arms,
extract result-context numeric claims, and record what changed.

Result-context = a number inside a tabular environment, or adjacent to \\% or
\\pm. Hyperparameters, equation constants and section numbers are not results,
and counting them would swamp the signal we care about.

Resumable: every finished paper is appended to pairs.jsonl, and a rerun skips
ids already present. arXiv is rate-limited, so a run that dies partway through
should not have to refetch.

Usage: python3 diff_arms.py [n_per_arm]
"""
import urllib.request, re, tarfile, io, gzip, json, os, sys, time, random
from collections import Counter

UA = {'User-Agent': 'ReviewDelta/1.0 (academic research; contact via paper)'}
# Probes import SLEEP by value at import time, so the delay is set from the
# environment rather than patched afterwards. run_all.py drops it once the
# source cache is warm, where the probes read from disk and a 4-second pause
# between local reads would add hours for nothing.
SLEEP = float(os.environ.get('REVIEWDELTA_SLEEP', '4.0'))
OUT = 'pairs.jsonl'

NUM = re.compile(r'(?<![\w.])(\d{1,4}\.\d{1,3})(?![\w])')
TABULAR = re.compile(r'\\begin\{tabular\}.*?\\end\{tabular\}', re.S)
PCT_PM = re.compile(r'(\d{1,4}\.\d{1,3})\s*(?:\\%|%|\\pm)')
COMMENT_LINE = re.compile(r'(?<!\\)%.*')

# Source cache. Every robustness probe (anchored_full, recall_probe,
# drop_taxonomy, baseline, longitudinal, claim_probe, fp_survival) calls
# fetch_source on the same papers this module already fetched, so without a
# cache a full rerun costs one network pass per probe. Caching the
# post-comment-strip .tex here means all callers share one pass, whichever
# import style they use.
#
# What is stored is the exact string fetch_source would have returned, so a
# cache hit and a cache miss produce identical extraction. Set
# REVIEWDELTA_CACHE='' to disable.
CACHE_DIR = os.environ.get('REVIEWDELTA_CACHE', '.source_cache')


def _cache_path(arxiv_id):
    # Shard by id prefix: one flat directory with 8k+ entries is slow on NFS,
    # which is where cluster scratch usually lives.
    safe = re.sub(r'[^\w.\-]', '_', arxiv_id)
    return os.path.join(CACHE_DIR, safe[:4], safe + '.tex.gz')


def cache_read(arxiv_id):
    if not CACHE_DIR:
        return None
    p = _cache_path(arxiv_id)
    if not os.path.exists(p):
        return None
    try:
        # Binary, not text mode: text mode applies universal-newline translation
        # and would silently turn a source file's \r\n into \n, so a cache hit
        # would not be byte-identical to the fetch it replaces.
        with gzip.open(p, 'rb') as fh:
            return fh.read().decode('utf-8')
    except Exception:
        # A truncated entry (killed mid-write) must not poison the run.
        try:
            os.remove(p)
        except OSError:
            pass
        return None


def cache_write(arxiv_id, tex):
    if not CACHE_DIR or not tex:
        return
    p = _cache_path(arxiv_id)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + f'.tmp{os.getpid()}'
    try:
        # Write-then-rename so concurrent shards never read a partial file.
        with gzip.open(tmp, 'wb') as fh:
            fh.write(tex.encode('utf-8'))
        os.replace(tmp, p)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass


def fetch_source(arxiv_id, tries=3):
    """Concatenated .tex for one arXiv version, or '' if unavailable."""
    hit = cache_read(arxiv_id)
    if hit is not None:
        return hit
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    raw = None
    delay = 15
    for t in range(tries):
        try:
            raw = urllib.request.urlopen(
                urllib.request.Request(url, headers=UA), timeout=120).read()
            break
        except Exception as e:
            code = getattr(e, 'code', None)
            if code == 404:
                return ''
            print(f"    retry {t+1}: {code or e}", file=sys.stderr)
            time.sleep(delay)
            delay *= 2
    if raw is None:
        return ''
    tex = ''
    try:
        tf = tarfile.open(fileobj=io.BytesIO(raw))
        for m in tf.getmembers():
            if m.name.lower().endswith('.tex') and m.size < 5_000_000:
                f = tf.extractfile(m)
                if f is not None:
                    tex += f.read().decode('utf-8', 'replace') + '\n'
    except tarfile.ReadError:
        try:
            tex = gzip.decompress(raw).decode('utf-8', 'replace')
        except Exception:
            return ''
    # Commented-out LaTeX is not a reported result. Leaving it in makes
    # "dropped" fire when an author merely uncomments an old table.
    tex = COMMENT_LINE.sub('', tex)
    cache_write(arxiv_id, tex)
    return tex


def result_numbers(tex):
    """Each number token counts once: tabular decimals, plus percent/plus-minus
    matches that sit outside every tabular block. Counting the percent pass over
    in-table text too would score one reported number as two."""
    out = Counter()
    spans = []
    for m in TABULAR.finditer(tex):
        spans.append(m.span())
        for n in NUM.findall(m.group(0)):
            out[n] += 1
    for m in PCT_PM.finditer(tex):
        if not any(a <= m.start() < b for a, b in spans):
            out[m.group(1)] += 1
    return out


def diff(rec):
    base, last = rec['id'], rec['version']
    a = fetch_source(f"{base}v1")
    time.sleep(SLEEP)
    b = fetch_source(f"{base}v{last}")
    time.sleep(SLEEP)
    if not a or not b:
        return {'id': base, 'arm': rec['arm'], 'status': 'source-unavailable'}
    A, B = result_numbers(a), result_numbers(b)
    shared = sum((A & B).values())
    dropped = sum((A - B).values())
    added = sum((B - A).values())
    total = shared + dropped + added
    return {
        'id': base, 'arm': rec['arm'], 'cat': rec['cat'], 'v_last': last,
        'comment': rec['comment'][:200], 'status': 'ok',
        'n_v1': sum(A.values()), 'n_vlast': sum(B.values()),
        'shared': shared, 'dropped': dropped, 'added': added,
        # Of the numbers v1 reported, what share did the final version not carry?
        'dropped_share': round(dropped / max(sum(A.values()), 1), 4),
        'churn_share': round(dropped / total, 4) if total else 0.0,
    }


if __name__ == '__main__':
    n_per_arm = int(sys.argv[1]) if len(sys.argv) > 1 else 70
    revised = json.load(open('arms.json'))['revised']
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                done.add(json.loads(line)['id'])
            except Exception:
                pass
    rng = random.Random(0)
    todo = []
    for arm in ('treatment', 'control'):
        pool = [r for r in revised if r['arm'] == arm and r['id'] not in done]
        rng.shuffle(pool)
        todo += pool[:n_per_arm]
    rng.shuffle(todo)
    print(f"{len(done)} already done; fetching {len(todo)} papers", file=sys.stderr)
    with open(OUT, 'a') as fh:
        for i, rec in enumerate(todo, 1):
            r = diff(rec)
            fh.write(json.dumps(r) + '\n')
            fh.flush()
            print(f"[{i}/{len(todo)}] {r['id']} {r['arm']} {r['status']} "
                  f"drop={r.get('dropped_share', '-')}", file=sys.stderr)
