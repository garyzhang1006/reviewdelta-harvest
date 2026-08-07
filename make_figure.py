#!/usr/bin/env python3
"""Figure 1: per-paper churn distribution under both extraction rules.

Reads pairs.jsonl and pairs_clean.jsonl, writes fig_bimodal.pdf. The exact-zero
papers get their own bar so the histogram's first bin cannot blur the paper's
central claim (half of papers change nothing) into the small-but-nonzero mass.
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

fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.7))
for ax, rows, title in [(axes[0], ok, 'Both rules'), (axes[1], okc, 'Clean rule only')]:
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
