#!/usr/bin/env python3
"""Stage 8 (validation): for treatment papers whose comment names an OpenReview-hosted venue
(ICLR/NeurIPS/COLM), fetch the arXiv title, then search OpenReview for a note
with that title and an acceptance signal. Writes orv_probe.json."""
import json, re, sys, time, html
import urllib.request

UA = {'User-Agent': 'ReviewDelta/1.0 (academic research; contact via paper)'}
SLEEP = 3.5

def get(url, tries=3):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            return urllib.request.urlopen(req, timeout=30).read().decode('utf-8', 'replace')
        except Exception as e:
            if t == tries - 1:
                print(f"  FAIL {url}: {e}", file=sys.stderr)
                return None
            time.sleep(8)

def arxiv_title(aid):
    page = get(f"https://arxiv.org/abs/{aid}")
    if not page:
        return None
    m = re.search(r'<title>\[[^\]]+\]\s*(.*?)</title>', page, re.S)
    return html.unescape(m.group(1)).strip() if m else None

def norm(t):
    return re.sub(r'[^a-z0-9]+', ' ', t.lower()).strip()

def openreview_search(title):
    q = urllib.parse.quote(title)
    url = f"https://api2.openreview.net/notes/search?term={q}&limit=10&content=title"
    raw = get(url)
    if raw is None:
        return []
    try:
        return json.loads(raw).get('notes', [])
    except Exception:
        return []

import urllib.parse
arms = json.load(open('arms.json'))['revised']
cands = [a for a in arms if a['arm'] == 'treatment'
         and re.search(r'ICLR|NeurIPS|COLM', a.get('comment', ''), re.I)]
print(f"{len(cands)} candidates", file=sys.stderr)

out = []
for i, a in enumerate(cands, 1):
    title = arxiv_title(a['id'])
    time.sleep(SLEEP)
    rec = {'id': a['id'], 'comment': a['comment'][:120], 'title': title,
           'or_match': None, 'or_venue': None}
    if title:
        notes = openreview_search(title)
        time.sleep(1.0)
        for n in notes:
            c = n.get('content', {})
            nt = c.get('title', {})
            nt = nt.get('value', '') if isinstance(nt, dict) else nt
            if norm(nt) == norm(title):
                ven = c.get('venue', {})
                ven = ven.get('value', '') if isinstance(ven, dict) else ven
                rec['or_match'] = n.get('id')
                rec['or_venue'] = ven
                break
    out.append(rec)
    print(f"[{i}/{len(cands)}] {a['id']} -> {rec['or_venue']}", file=sys.stderr)

json.dump(out, open('orv_probe.json', 'w'), indent=1)
hit = sum(1 for r in out if r['or_match'])
print(f"matched {hit}/{len(out)}")
