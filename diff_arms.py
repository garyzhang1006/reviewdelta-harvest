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
SLEEP = 4.0
OUT = 'pairs.jsonl'

NUM = re.compile(r'(?<![\w.])(\d{1,4}\.\d{1,3})(?![\w])')
TABULAR = re.compile(r'\\begin\{tabular\}.*?\\end\{tabular\}', re.S)
PCT_PM = re.compile(r'(\d{1,4}\.\d{1,3})\s*(?:\\%|%|\\pm)')
COMMENT_LINE = re.compile(r'(?<!\\)%.*')


def fetch_source(arxiv_id, tries=3):
    """Concatenated .tex for one arXiv version, or '' if unavailable."""
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
    return COMMENT_LINE.sub('', tex)


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
