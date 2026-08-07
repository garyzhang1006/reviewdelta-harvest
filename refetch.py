#!/usr/bin/env python3
"""One-off: rebuild pairs.jsonl with the deduplicated result_numbers on the
same 272 papers, then rebuild longitudinal.jsonl. Resumable like the originals."""
import json, os, sys, subprocess
import diff_arms as da

arms = {r['id']: r for r in json.load(open('arms.json'))['revised']}
manifest = json.load(open('refetch_manifest.json'))
for r in manifest:
    if not r.get('version'):
        r['version'] = arms[r['id']]['version']

done = set()
if os.path.exists('pairs.jsonl'):
    for line in open('pairs.jsonl'):
        try:
            done.add(json.loads(line)['id'])
        except Exception:
            pass
todo = [r for r in manifest if r['id'] not in done]
print(f"pairs: {len(done)} done, {len(todo)} to go", file=sys.stderr)
with open('pairs.jsonl', 'a') as fh:
    for i, rec in enumerate(todo, 1):
        r = da.diff(rec)
        fh.write(json.dumps(r) + '\n')
        fh.flush()
        print(f"[{i}/{len(todo)}] {r['id']} {r['arm']} {r['status']} "
              f"drop={r.get('dropped_share', '-')}", file=sys.stderr)
print("pairs done; starting longitudinal", file=sys.stderr)
subprocess.run([sys.executable, 'longitudinal.py'], check=True)
