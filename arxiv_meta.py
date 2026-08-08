#!/usr/bin/env python3
"""
Stage 1b: harvest arXiv metadata in bulk through the export API.

Replaces harvest.py's per-paper /abs/ scraping. harvest.py fetches one HTML page
per paper, so a 10,071-candidate frame costs 10,071 requests and about 8 hours at
its 3-second delay. The API returns the same three fields we need (version count,
comment, primary category) in slices of up to 2,000 entries, which brings the same
frame down to roughly six requests.

harvest.py's docstring says the API "rate-limits to the point of being unusable".
That is true of per-id querying. This module never queries per id: it pages a
category-plus-date search and reads whole slices.

  version  <id> comes back as .../abs/2501.12345v3, so max version is the suffix.
  comment  <arxiv:comment> is the arm-label field harvest.VENUE matches against.
  cat      <arxiv:primary_category term="cs.LG"/>.

UNVERIFIED AS SHIPPED. Nobody has run this against the live API yet. Run
`python3 arxiv_meta.py --validate` first; it fetches one small slice and asserts
that versions, comments and categories all parse, then agrees with a handful of
harvest.abs_meta() results on the same ids. Do not launch the full harvest until
that passes.

Writes the same meta.jsonl records harvest.py writes, so downstream stages
(arms assignment, diff_arms, report.py) do not change.

Usage:
  python3 arxiv_meta.py --validate
  python3 arxiv_meta.py --months 2025-01 2025-03 --cats cs.LG cs.CL
  python3 arxiv_meta.py --all-2025            # every month of 2025, both cats
"""
import argparse
import calendar
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

import harvest as hv

API = "http://export.arxiv.org/api/query"
SLICE = 2000          # API hard cap per call
SLEEP = 3.0           # arXiv asks for 3s between calls
META = "meta.jsonl"

ENTRY = re.compile(r"<entry>(.*?)</entry>", re.S)
E_ID = re.compile(r"<id>\s*https?://arxiv\.org/abs/([^<\s]+?)\s*</id>", re.S)
E_COMMENT = re.compile(r"<arxiv:comment[^>]*>(.*?)</arxiv:comment>", re.S)
E_PRIMARY = re.compile(r'<arxiv:primary_category[^>]*\bterm="([^"]+)"')
E_TOTAL = re.compile(r"<opensearch:totalResults[^>]*>(\d+)</opensearch:totalResults>")


def _unxml(s):
    for a, b in (("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'),
                 ("&apos;", "'"), ("&amp;", "&")):
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s)).strip()


def _get(url, tries=4):
    delay = 15
    for t in range(tries):
        try:
            return urllib.request.urlopen(
                urllib.request.Request(url, headers=hv.UA), timeout=120
            ).read().decode("utf-8", "replace")
        except Exception as e:
            code = getattr(e, "code", None)
            print(f"    retry {t + 1}: {code or e}", file=sys.stderr)
            time.sleep(delay)
            delay *= 2
    return ""


def parse_entries(xml):
    """[{id, version, comment, cat}] for every <entry> in one API response."""
    out = []
    for body in ENTRY.findall(xml):
        m = E_ID.search(body)
        if not m:
            continue
        raw = m.group(1)
        if "v" in raw.rsplit("/", 1)[-1]:
            base, _, vs = raw.rpartition("v")
            version = int(vs) if vs.isdigit() else 1
        else:
            base, version = raw, 1
        c = E_COMMENT.search(body)
        p = E_PRIMARY.search(body)
        out.append({
            "id": base,
            "version": version,
            "comment": _unxml(c.group(1)) if c else "",
            "cat": p.group(1) if p else "",
        })
    return out


def month_bounds(month):
    """'2025-03' -> ('202503010000', '202503312359') in the API's date format."""
    y, m = (int(x) for x in month.split("-"))
    return f"{y}{m:02d}010000", f"{y}{m:02d}{calendar.monthrange(y, m)[1]:02d}2359"


def harvest_slice(cat, month, seen, cap=None):
    """Every entry arXiv lists for one category-month. Pages until exhausted.

    cap bounds the candidates taken per category-month, chronological head
    first, which is what the original listing-page harvest did with its
    show=1000 limit. Use cap=1000 to reproduce that frame's sampling on a
    session-limited machine; the full frame for the same months is roughly
    2.5x larger and costs a proportionally longer source fetch.
    """
    lo, hi = month_bounds(month)
    q = f"cat:{cat} AND submittedDate:[{lo} TO {hi}]"
    got, start, total = [], 0, None
    while True:
        if cap is not None and start >= cap:
            break
        take = SLICE if cap is None else min(SLICE, cap - start)
        url = (f"{API}?search_query={urllib.parse.quote(q)}"
               f"&start={start}&max_results={take}")
        xml = _get(url)
        if not xml:
            print(f"    {cat} {month}: empty response at start={start}",
                  file=sys.stderr)
            break
        if total is None:
            t = E_TOTAL.search(xml)
            total = int(t.group(1)) if t else 0
            print(f"  {cat} {month}: {total} total", file=sys.stderr)
        batch = parse_entries(xml)
        if not batch:
            break
        for r in batch:
            if r["id"] not in seen:
                seen.add(r["id"])
                got.append(r)
        start += len(batch)
        time.sleep(SLEEP)
        if start >= total:
            break
    return got


def validate():
    """One small slice, then cross-check against harvest.py's proven scraper.

    Exits nonzero on any mismatch. This is the gate: the API path is unverified
    until this passes on the machine that will run the harvest.
    """
    print("fetching one validation slice (cs.LG, 2025-01-01..02)", file=sys.stderr)
    lo, hi = "202501010000", "202501022359"
    q = f"cat:cs.LG AND submittedDate:[{lo} TO {hi}]"
    xml = _get(f"{API}?search_query={urllib.parse.quote(q)}&start=0&max_results=50")
    assert xml, "empty API response: the API is unreachable or blocking this host"
    rows = parse_entries(xml)
    assert rows, "parsed zero entries: the API response shape changed"

    print(f"  parsed {len(rows)} entries", file=sys.stderr)
    assert all(re.fullmatch(r"\d{4}\.\d{4,5}", r["id"]) for r in rows), \
        "id parse failed: version suffix not stripped cleanly"
    assert any(r["version"] >= 2 for r in rows), \
        "no revised papers parsed: version suffix is missing from <id>"
    assert any(r["cat"] for r in rows), "primary_category never parsed"
    assert any(r["comment"] for r in rows), "comment field never parsed"
    nrev = sum(1 for r in rows if r["version"] >= 2)
    ncom = sum(1 for r in rows if r["comment"])
    print(f"  versions ok ({nrev}/{len(rows)} revised), "
          f"comments ok ({ncom}/{len(rows)} non-empty)", file=sys.stderr)

    print("cross-checking 3 ids against harvest.abs_meta() ...", file=sys.stderr)
    bad = []
    for r in rows[:3]:
        time.sleep(SLEEP)
        ref = hv.abs_meta(r["id"])
        if ref is None:
            print(f"  {r['id']}: abs page unavailable, skipped", file=sys.stderr)
            continue
        if ref["version"] != r["version"]:
            bad.append(f"{r['id']} version api={r['version']} abs={ref['version']}")
        # The two paths rewrite comments differently: the abs page replaces
        # every link with the literal "this https URL" while the API keeps the
        # real address. Mask URLs on both sides before comparing, and treat the
        # arm label (what the comment actually feeds) as the decisive check.
        mask = lambda s: re.sub(r"\W+", "", re.sub(
            r"https?://\S+|this https url", " U ", s, flags=re.I)).lower()
        if mask(r["comment"]) != mask(ref["comment"]):
            if bool(hv.VENUE.search(r["comment"])) != bool(hv.VENUE.search(ref["comment"])):
                bad.append(f"{r['id']} arm-relevant comment mismatch "
                           f"api={r['comment'][:60]!r} abs={ref['comment'][:60]!r}")
            else:
                print(f"    {r['id']}: comment differs only in URL/whitespace "
                      "form; arm label agrees", file=sys.stderr)
        print(f"  {r['id']}: v{r['version']} vs v{ref['version']} "
              f"{'OK' if not bad else 'MISMATCH'}", file=sys.stderr)
    if bad:
        print("\nVALIDATION FAILED:", file=sys.stderr)
        for b in bad:
            print("  " + b, file=sys.stderr)
        print("\nFall back to the proven scraper: python3 harvest.py", file=sys.stderr)
        sys.exit(1)
    print("\nVALIDATION PASSED. Safe to run the full harvest.", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="check the API path against harvest.py, then exit")
    ap.add_argument("--cats", nargs="+", default=hv.CATS)
    ap.add_argument("--months", nargs="+", default=hv.MONTHS)
    ap.add_argument("--all-2025", action="store_true",
                    help="every month of 2025 instead of --months")
    ap.add_argument("--cap", type=int, default=None,
                    help="max candidates per category-month (1000 reproduces "
                         "the original listing-page frame)")
    ap.add_argument("--out", default=META)
    a = ap.parse_args()

    if a.validate:
        validate()
        return

    months = [f"2025-{m:02d}" for m in range(1, 13)] if a.all_2025 else a.months

    # Resume: never refetch metadata we already hold.
    have = {}
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line)
                have[r["id"]] = r
            except Exception:
                pass
    print(f"{len(have)} metadata records already cached", file=sys.stderr)

    seen = set(have)
    nnew = 0
    with open(a.out, "a") as fh:
        for cat in a.cats:
            for mo in months:
                for r in harvest_slice(cat, mo, seen, cap=a.cap):
                    fh.write(json.dumps(r) + "\n")
                    nnew += 1
                fh.flush()
                print(f"  running total: {len(have) + nnew}", file=sys.stderr)

    for line in open(a.out):
        try:
            r = json.loads(line)
            have[r["id"]] = r
        except Exception:
            pass

    revised = [r for r in have.values() if r["version"] >= 2]
    for r in revised:
        r["arm"] = "treatment" if hv.VENUE.search(r["comment"]) else "control"
    t = sum(1 for r in revised if r["arm"] == "treatment")
    print(f"\nmetadata: {len(have)} papers ({nnew} new this run)")
    print(f"revised (v>=2): {len(revised)} ({len(revised) / max(len(have), 1):.0%})")
    print(f"  treatment: {t}")
    print(f"  control  : {len(revised) - t}")
    json.dump({"all": list(have.values()), "revised": revised},
              open("arms.json", "w"), indent=1)
    print("wrote arms.json")


if __name__ == "__main__":
    main()
