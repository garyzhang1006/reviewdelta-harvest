#!/usr/bin/env python3
"""Figure 1: per-paper churn distribution under three matchers.

Reads pairs.jsonl, pairs_clean.jsonl and anchored_full.jsonl, writes
fig_bimodal.pdf. The exact-zero papers get their own bar so the histogram's
first bin cannot blur the paper's central claim (half of papers change
nothing) into the small-but-nonzero mass. The anchored panel shows the
strictest matcher, where restructuring counts as churn, so the reader sees
the 37% zero share next to the 53% instead of meeting it only in Table 2.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(path):
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows if r['status'] == 'ok' and r['n_v1'] > 0]


ok, okc = load('pairs.jsonl'), load('pairs_clean.jsonl')
anch = [r for r in (json.loads(l) for l in open('anchored_full.jsonl'))
        if r['status'] == 'ok' and r['anchored_total'] >= 5]
for r in anch:
    r['dropped_share'] = r['anchored_share']

fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.1))
for ax, rows, title in [(axes[0], ok, 'Both rules'), (axes[1], okc, 'Clean rule only'),
                        (axes[2], anch, 'Anchored (position-strict)')]:
    d = np.array([r['dropped_share'] for r in rows])
    nz = d[d > 0]
    bins = np.linspace(0, 1, 21)
    ax.bar(-0.07, (d == 0).sum(), width=0.05, color='#2d4a63', label='exactly zero')
    ax.hist(nz, bins=bins, color='#7aa6c9', edgecolor='white', linewidth=0.6, label='nonzero')
    ax.axvline(-0.02, color='gray', linewidth=0.5, linestyle=':')
    ax.set_title(f'{title} (n={len(d)})', fontsize=9)
    ax.set_xlabel('Share of v1 result numbers not carried forward', fontsize=8)
    ax.set_xlim(-0.12, 1.02)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
axes[0].set_ylabel('Papers', fontsize=8)
axes[0].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig('fig_bimodal.pdf', bbox_inches='tight')
print('wrote fig_bimodal.pdf')
