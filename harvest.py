#!/usr/bin/env python3
"""
Stage 1: harvest arXiv metadata and assign treatment/control arms.

Treatment = revised (v>=2) AND the comment field names an accepting venue.
Control   = revised (v>=2) AND no venue signal.

The arms are the whole design. A v1-to-vN diff on its own measures revision,
not review: typo passes, added baselines and pre-review polish all land in it.
Venue signals undercount review, since plenty of accepted papers never touch
their comment field. That mislabels treatment papers as controls and pushes
the arm difference toward zero, so it costs power rather than manufacturing
an effect.

We read listing pages for ids and abs pages for version count and comment.
The export.arxiv.org API returns the same fields in bulk but rate-limits to
the point of being unusable for a few hundred papers; arxiv.org itself serves
these pages reliably at one request every few seconds.

Resumable: abs results append to meta.jsonl and a rerun skips known ids.

Usage: python3 harvest.py [n_abs_to_fetch]
"""
import urllib.request, re, time, json, sys, os, random

UA = {'User-Agent': 'ReviewDelta/1.0 (academic research; contact via paper)'}
SLEEP = 3.0
META = 'meta.jsonl'
MONTHS = ['2025-01', '2025-03', '2025-05', '2025-07', '2025-09']
CATS = ['cs.LG', 'cs.CL']

VENUE = re.compile(
    r'\b(accepted|to appear|camera[- ]ready|published|presented)\b.{0,60}?'
    r'\b(neurips|nips|icml|iclr|aaai|ijcai|acl|emnlp|naacl|cvpr|iccv|eccv|'
    r'kdd|www|sigir|colm|tmlr|jmlr|aistats|uai|coling|eacl|wacv|bmvc|corl|'
    r'interspeech|icassp|miccai|chi|conference|workshop|journal|transactions)\b',
    re.I | re.S)


def get(url, tries=4):
    delay = 15
    for t in range(tries):
        try:
            return urllib.request.urlopen(
                urllib.request.Request(url, headers=UA), timeout=90).read().decode(
                    'utf-8', 'replace')
        except Exception as e:
            code = getattr(e, 'code', None)
            if code == 404:
                return ''
            print(f"    retry {t+1}: {code or e}", file=sys.stderr)
            time.sleep(delay)
            delay *= 2
    return ''


def listing_ids(cat, month, show=1000):
    html = get(f"https://arxiv.org/list/{cat}/{month}?skip=0&show={show}")
    ids = re.findall(r'arXiv:(\d{4}\.\d{4,5})', html)
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


ABS_V = re.compile(r'\[v(\d+)\]')
ABS_C = re.compile(r'comments mathjax">(.*?)</td>', re.S)
ABS_SUBJ = re.compile(r'primary-subject">(.*?)</span>', re.S)


def abs_meta(aid):
    html = get(f"https://arxiv.org/abs/{aid}")
    if not html:
        return None
    vs = [int(v) for v in ABS_V.findall(html)]
    c = ABS_C.search(html)
    comment = re.sub(r'<[^>]+>', '', c.group(1)) if c else ''
    comment = re.sub(r'\s+', ' ', comment).strip()
    subj = ABS_SUBJ.search(html)
    return {
        'id': aid,
        'version': max(vs) if vs else 1,
        'comment': comment,
        'cat': re.sub(r'<[^>]+>', '', subj.group(1)).strip() if subj else '',
    }


if __name__ == '__main__':
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 700

    done = {}
    if os.path.exists(META):
        for line in open(META):
            try:
                r = json.loads(line)
                done[r['id']] = r
            except Exception:
                pass
    print(f"{len(done)} abs records already cached", file=sys.stderr)

    cands = []
    for cat in CATS:
        for mo in MONTHS:
            ids = listing_ids(cat, mo)
            print(f"  listing {cat} {mo}: {len(ids)} ids", file=sys.stderr)
            cands += ids
            time.sleep(SLEEP)
    seen, uniq = set(), []
    for i in cands:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    print(f"{len(uniq)} unique candidate ids", file=sys.stderr)

    todo = [i for i in uniq if i not in done]
    random.Random(0).shuffle(todo)
    todo = todo[:budget]
    print(f"fetching {len(todo)} abs pages", file=sys.stderr)

    with open(META, 'a') as fh:
        for n, aid in enumerate(todo, 1):
            m = abs_meta(aid)
            if m:
                fh.write(json.dumps(m) + '\n')
                fh.flush()
                done[aid] = m
            if n % 25 == 0:
                rev = sum(1 for r in done.values() if r['version'] >= 2)
                print(f"  [{n}/{len(todo)}] cached {len(done)}, revised {rev}",
                      file=sys.stderr)
            time.sleep(SLEEP)

    recs = list(done.values())
    revised = [r for r in recs if r['version'] >= 2]
    for r in revised:
        r['arm'] = 'treatment' if VENUE.search(r['comment']) else 'control'
    t = sum(1 for r in revised if r['arm'] == 'treatment')
    print(f"\npapers with metadata: {len(recs)}")
    print(f"revised (v>=2): {len(revised)} ({len(revised)/max(len(recs),1):.0%})")
    print(f"  treatment (venue signal): {t} ({t/max(len(recs),1):.0%} of all)")
    print(f"  control   (no signal)   : {len(revised)-t}")
    json.dump({'all': recs, 'revised': revised}, open('arms.json', 'w'), indent=1)
    print("wrote arms.json")
