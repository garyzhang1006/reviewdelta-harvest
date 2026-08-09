#!/usr/bin/env python3
"""
Print every number that appears in the paper, from the released data files.

Reads pairs.jsonl (both extraction rules), pairs_clean.jsonl (percent/plus-minus
only) and precision.json. Runs in seconds; no network access.
"""
import json, re, os
import numpy as np

rng = np.random.default_rng(0)


def load(path):
    rows = [json.loads(l) for l in open(path)]
    return rows, [r for r in rows if r['status'] == 'ok' and r['n_v1'] > 0]


def perm_p(t, c, reps=20000, seed=0):
    """Seeded per call, so a test's p-value does not depend on call order."""
    r = np.random.default_rng(seed)
    obs = t.mean() - c.mean()
    allv = np.concatenate([t, c])
    null = np.array([(lambda q: q[:len(t)].mean() - q[len(t):].mean())(r.permutation(allv))
                     for _ in range(reps)])
    return obs, (np.abs(null) >= abs(obs)).mean()


def arms(rows):
    return (np.array([r['dropped_share'] for r in rows if r['arm'] == 'treatment']),
            np.array([r['dropped_share'] for r in rows if r['arm'] == 'control']))


def shape(rows, label):
    d = np.array([r['dropped_share'] for r in rows])
    tot = sum(r['n_v1'] for r in rows)
    dr = sum(r['dropped'] for r in rows)
    print(f"{label}: n={len(rows)}  zero={(d==0).mean():.0%}  "
          f"1-20%={((d>0)&(d<=0.2)).mean():.0%}  >20%={(d>0.2).mean():.0%}  "
          f"pooled={dr/tot:.1%} of {tot:,}  mean={d.mean():.3f} median={np.median(d):.3f}")


print("== Section 2: corpus ==")
meta = [json.loads(l) for l in open('meta.jsonl')]
rev = [m for m in meta if m['version'] >= 2]
arms_json = json.load(open('arms.json'))['revised']
nt = sum(1 for r in arms_json if r['arm'] == 'treatment')
print(f"metadata on {len(meta)} papers, revised {len(rev)} ({len(rev)/len(meta):.0%}), "
      f"treatment {nt}, control {len(arms_json)-nt}")
raw, ok = load('pairs.jsonl')
print(f"attempted {len(raw)}: {sum(1 for r in raw if r['status']!='ok')} source-unavailable, "
      f"{sum(1 for r in raw if r['status']=='ok' and r['n_v1']==0)} with no v1 numbers, "
      f"{len(ok)} usable")
t, c = arms(ok)
print(f"usable arms: treatment {len(t)}, control {len(c)}")

print("\n== Section 3: precision ==")
if os.path.exists('precision.json'):
    p = json.load(open('precision.json'))
    print(f"hand-labeled {p['n']}, precision {p['precision']:.1%}")
    for k, v in sorted(p['false_positives'].items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(v):2d}  {k}")
    smp = json.load(open('precision_sample.json'))
    fp_idx = {i for v in p['false_positives'].values() for i in v}
    bypaper = {}
    for i, rec in enumerate(smp, 1):
        bypaper.setdefault(rec['paper'], []).append(0 if i in fp_idx else 1)
    papers_ = list(bypaper.values())
    rP = np.random.default_rng(0)
    prec_b = [np.mean([x for k_ in rP.integers(0, len(papers_), len(papers_))
                       for x in papers_[k_]]) for _ in range(20000)]
    print(f"  clustered 95% CI over {len(papers_)} papers: "
          f"[{np.percentile(prec_b, 2.5):.0%}, {np.percentile(prec_b, 97.5):.0%}]")

    if os.path.exists('precision_blind.json'):
        bl = json.load(open('precision_blind.json'))
        bset = {x['index'] for x in bl['not_result']}
        aset = fp_idx
        n_ = 150
        po = sum(1 for i in range(1, n_+1) if (i in bset) == (i in aset)) / n_
        pe = ((n_-len(bset))/n_)*((n_-len(aset))/n_) + (len(bset)/n_)*(len(aset)/n_)
        print(f"blind second annotator: flags {len(bset)}, agreement {po:.1%}, "
              f"kappa {(po-pe)/(1-pe):.2f}, implied precision {(n_-len(bset))/n_:.1%}")

if os.path.exists('precision2.json'):
    p2 = json.load(open('precision2.json'))
    print(f"tranche-2 hand-labeled {p2['n']} (papers beyond the first 272 attempts, "
          f"{p2['papers']} papers), precision {p2['precision']:.1%}, "
          f"clustered CI [{p2['clustered_ci'][0]:.2f}, {p2['clustered_ci'][1]:.2f}]")
    if os.path.exists('precision2_blind.json'):
        bl2 = json.load(open('precision2_blind.json'))
        b2 = {x['index'] for x in bl2['not_result']}
        a2 = {i for v in p2['not_result'].values() for i in v}
        n2 = p2['n']
        po2 = sum(1 for i in range(1, n2+1) if (i in b2) == (i in a2)) / n2
        pe2 = ((n2-len(b2))/n2)*((n2-len(a2))/n2) + (len(b2)/n2)*(len(a2)/n2)
        print(f"tranche-2 blind annotator: flags {len(b2)}, agreement {po2:.1%}, "
              f"kappa {(po2-pe2)/(1-pe2):.2f}")

if os.path.exists('claim_probe.json'):
    import collections as _c
    pr = json.load(open('claim_probe.json'))
    both = [(r['doc_share'], 1 - r['anchored_kept']/r['anchored_total'])
            for r in pr if r['anchored_total'] >= 5]
    d_ = np.array([x[0] for x in both]); a_ = np.array([x[1] for x in both])
    bysize = sorted(pr, key=lambda r: r['n_v1'])
    coll = []
    for i, r in enumerate(bysize):
        j = i+1 if i+1 < len(bysize) else i-1
        other = _c.Counter(bysize[j]['vn_numbers'])
        mine = _c.Counter(r['v1_numbers'])
        coll.append(sum((mine & other).values()) / max(sum(mine.values()), 1))
    print(f"claim probe (n={len(pr)}, {len(both)} with >=5 anchored cells): "
          f"doc mean {d_.mean():.2f}, anchored mean {a_.mean():.2f}, "
          f"median gap {np.median(a_-d_):+.2f}; "
          f"collision baseline mean {np.mean(coll):.1%} median {np.median(coll):.1%}")
    sub = [r for r in pr if r['anchored_total'] >= 5]
    pool_anch = 1 - sum(r['anchored_kept'] for r in sub) / sum(r['anchored_total'] for r in sub)
    pool_doc = sum(r['doc_share'] * r['n_v1'] for r in sub) / sum(r['n_v1'] for r in sub)
    print(f"  pooled on the subsample: doc {pool_doc:.1%}, anchored {pool_anch:.1%}, "
          f"gap {pool_anch - pool_doc:+.1%}")

print("\n== Section 4: shape ==")
shape(ok, "both rules")
rawc, okc = load('pairs_clean.jsonl')
shape(okc, "clean rule")
print(f"clean rule numbers per paper: median {np.median([r['n_v1'] for r in okc]):g}, "
      f"both rules: median {np.median([r['n_v1'] for r in ok]):g}")
hv = max(okc, key=lambda r: r['n_v1'])
tot_c = sum(r['n_v1'] for r in okc)
dr_c = sum(r['dropped'] for r in okc)
_without = (dr_c - hv['dropped']) / (tot_c - hv['n_v1'])
print(f"clean-rule outlier: one paper with {hv['n_v1']:,} numbers at {hv['dropped_share']:.0%} churn; "
      f"pooled clean {dr_c/tot_c:.1%} -> {_without:.1%} without it "
      f"({dr_c/tot_c - _without:.1%} points of the pooled figure)")

ad = sum(r['added'] for r in ok)
dr = sum(r['dropped'] for r in ok)
both0 = sum(1 for r in ok if r['added'] == 0 and r['dropped'] == 0)
print(f"added vs dropped (pooled): {ad:,} vs {dr:,} (ratio {ad/dr:.2f}); "
      f"pairs untouched in both directions: {both0/len(ok):.0%}")
top = sorted(ok, key=lambda r: -r['dropped'])
cum = np.cumsum([r['dropped'] for r in top])
print(f"papers supplying half the dropped numbers: {int(np.searchsorted(cum, dr/2)) + 1} of {len(ok)}")
d4 = np.array([r['dropped_share'] for r in ok])
n4 = np.array([r['n_v1'] for r in ok])
ch4 = d4[d4 > 0]
mid_n = ((d4 > 0) & (d4 <= 0.2)).sum(); hi_n = (d4 > 0.2).sum()
print(f"changers: n={len(ch4)}, median dropped share {np.median(ch4):.2f}")
print(f"density per unit share: (0,0.2] {mid_n/0.2:.0f} vs (0.2,1] {hi_n/0.8:.0f} papers/unit")
print(f"total-replacement papers: {(d4 == 1).sum()}, their median v1 count "
      f"{np.median(n4[d4 == 1]):.1f} vs corpus {np.median(n4):.0f}")
dc20 = np.array([r['dropped_share'] for r in okc if r['n_v1'] >= 20])
print(f"clean rule >=20 numbers (n={len(dc20)}): zero {(dc20 == 0).mean():.0%} "
      f"mid {((dc20 > 0) & (dc20 <= 0.2)).mean():.0%} hi {(dc20 > 0.2).mean():.0%}")
small5 = sum(1 for r in okc if r['n_v1'] <= 5)
hi_small5 = sum(1 for r in okc if r['n_v1'] <= 5 and r['dropped_share'] > 0.2)
hi_allc = sum(1 for r in okc if r['dropped_share'] > 0.2)
print(f"clean rule papers with <=5 numbers: {small5}, supplying {hi_small5} of "
      f"{hi_allc} past-20% papers")
w_t, w_c = 155/883, 728/883
tr_ = [r for r in ok if r['arm'] == 'treatment']
co_ = [r for r in ok if r['arm'] == 'control']
def _w(f):
    return w_t*np.mean([f(r) for r in tr_]) + w_c*np.mean([f(r) for r in co_])
pool_t = sum(r['dropped'] for r in tr_) / sum(r['n_v1'] for r in tr_)
pool_c = sum(r['dropped'] for r in co_) / sum(r['n_v1'] for r in co_)
print(f"reweighted to population arm shares ({155/883:.1%} treatment): "
      f"zero {_w(lambda r: r['dropped_share'] == 0):.1%}, "
      f">20% {_w(lambda r: r['dropped_share'] > 0.2):.1%}, "
      f"pooled {w_t*pool_t + w_c*pool_c:.1%}")

print("\n== Section 5: arm comparison ==")
WORKSHOP = re.compile(r'workshop', re.I)
specs = [
    ("both rules, all", ok),
    ("both rules, main-conf treatment", [r for r in ok if not (
        r['arm'] == 'treatment' and WORKSHOP.search(r.get('comment', '')))]),
    ("both rules, >=10 numbers", [r for r in ok if r['n_v1'] >= 10]),
    ("both rules, >=100 numbers", [r for r in ok if r['n_v1'] >= 100]),
    ("both rules, cs.LG only", [r for r in ok if 'cs.LG' in r.get('cat', '')]),
    ("both rules, cs.CL only", [r for r in ok if 'cs.CL' in r.get('cat', '')]),
    ("controls w/o venue mention", [r for r in ok if r['arm'] == 'treatment' or not
        re.search(r'NeurIPS|ICML|ICLR|ACL\b|EMNLP|NAACL|CVPR|ICCV|ECCV|AAAI|IJCAI|COLM|'
                  r'COLING|AISTATS|KDD|SIGIR|CIKM|WSDM|TMLR|Findings|Spotlight|'
                  r'Main Conference|camera[- ]ready|[Aa]ccepted', r.get('comment', ''), re.I)]),
    ("clean rule, all", okc),
    ("clean rule, >=5 numbers", [r for r in okc if r['n_v1'] >= 5]),
    ("clean rule, >=20 numbers", [r for r in okc if r['n_v1'] >= 20]),
]
for label, rows in specs:
    a, b = arms(rows)
    if len(a) < 5 or len(b) < 5:
        print(f"{label:34s} too few (t={len(a)}, c={len(b)})")
        continue
    obs, p = perm_p(a, b)
    print(f"{label:34s} t={len(a):2d} c={len(b):2d}  {a.mean():.3f} vs {b.mean():.3f}  "
          f"diff {obs:+.3f}  p={p:.2f}")

t, c = arms(ok)
comb = np.concatenate([t, c])
order = comb.argsort()
ranks = np.empty(len(comb), float)
ranks[order] = np.arange(1, len(comb) + 1)
for v in np.unique(comb):
    m = comb == v
    ranks[m] = ranks[m].mean()
u = ranks[:len(t)].mean() - ranks[len(t):].mean()
r2 = np.random.default_rng(0)
nr = np.array([(lambda q: q[:len(t)].mean() - q[len(t):].mean())(r2.permutation(ranks))
               for _ in range(20000)])
print(f"rank test (both rules): p = {(np.abs(nr) >= abs(u)).mean():.2f}")
r3 = np.random.default_rng(0)
boot = np.array([r3.choice(t, len(t), True).mean() - r3.choice(c, len(c), True).mean()
                 for _ in range(20000)])
print(f"bootstrap 95% CI: [{np.percentile(boot,2.5):+.3f}, {np.percentile(boot,97.5):+.3f}]")

tv = [r['v_last'] for r in ok if r['arm'] == 'treatment']
cv = [r['v_last'] for r in ok if r['arm'] == 'control']
obs = np.mean(tv) - np.mean(cv)
allv = np.array(tv + cv)
r4 = np.random.default_rng(0)
nv = np.array([(lambda q: q[:len(tv)].mean() - q[len(tv):].mean())(r4.permutation(allv))
               for _ in range(20000)])
print(f"version-count balance: {np.mean(tv):.2f} vs {np.mean(cv):.2f}, "
      f"diff {obs:+.2f}, p = {(np.abs(nv) >= abs(obs)).mean():.2f}")

se = np.sqrt(t.var(ddof=1)/len(t) + c.var(ddof=1)/len(c))
MDE = se*(1.96+0.84)
print(f"arm-test MDE (80% power, two-sided .05): {MDE:.3f}")
d_all = np.array([r['dropped_share'] for r in ok])
dr_ = np.array([r['dropped'] for r in ok]); nv_ = np.array([r['n_v1'] for r in ok])
rB = np.random.default_rng(0); idxB = np.arange(len(ok))
P, Z, T = [], [], []
for _ in range(20000):
    p = rB.choice(idxB, len(ok), True)
    P.append(dr_[p].sum()/nv_[p].sum()); Z.append((d_all[p]==0).mean()); T.append((d_all[p]>0.2).mean())
print(f"cluster CIs: pooled [{np.percentile(P,2.5):.1%},{np.percentile(P,97.5):.1%}]  "
      f"zero [{np.percentile(Z,2.5):.0%},{np.percentile(Z,97.5):.0%}]  "
      f">20% [{np.percentile(T,2.5):.0%},{np.percentile(T,97.5):.0%}]")

print("\n== Section 4 addendum: depth among changers ==")
d = np.array([r['dropped_share'] for r in ok])
hi = d[d > 0.2]
print(f"papers >20%: n={len(hi)}, median share replaced {np.median(hi):.2f}, "
      f"share replacing over half {(hi > 0.5).mean():.0%}")

print("\n== Section 5 addendum: diagnosing the rule disagreement ==")
okc_ids = {r['id'] for r in okc}
shared = [r for r in ok if r['id'] in okc_ids]
a, b = arms(shared)
obs, p = perm_p(a, b)
print(f"both-rules metric on the clean rule's papers (n={len(shared)}): "
      f"{a.mean():.3f} vs {b.mean():.3f}  diff {obs:+.3f}  p={p:.2f}")
okb = {r['id']: r for r in ok}
recs = []
for r in okc:
    B = okb.get(r['id'])
    if B is None:
        continue
    n_tab = B['n_v1'] - r['n_v1']
    if n_tab > 0:
        recs.append({'arm': B['arm'],
                     'dropped_share': max(0.0, (B['dropped'] - r['dropped']) / n_tab)})
a = np.array([x['dropped_share'] for x in recs if x['arm'] == 'treatment'])
b = np.array([x['dropped_share'] for x in recs if x['arm'] == 'control'])
obs, p = perm_p(a, b)
print(f"tabular-only numbers, same papers (t={len(a)}, c={len(b)}): "
      f"{a.mean():.3f} vs {b.mean():.3f}  diff {obs:+.3f}  p={p:.2f}")
import numpy as _np
pairsx = _np.array([(okb[r['id']]['dropped_share'], r['dropped_share'])
                    for r in okc if r['id'] in okb])
print(f"per-paper correlation between metrics: r = "
      f"{_np.corrcoef(pairsx[:,0], pairsx[:,1])[0,1]:.2f}")

print("\n== Section 5: the first-tranche disagreement (historical) ==")
first = [json.loads(l) for l in open('pairs.jsonl')][:112]
fok = [r for r in first if r['status'] == 'ok' and r['n_v1'] > 0]
a, b = arms(fok)
obs, p = perm_p(a, b)
print(f"first 112 attempts, both rules (t={len(a)}, c={len(b)}): "
      f"{a.mean():.3f} vs {b.mean():.3f}  diff {obs:+.3f}  p={p:.2f}")
firstc = [json.loads(l) for l in open('pairs_clean.jsonl')][:107]
fokc = [r for r in firstc if r['status'] == 'ok' and r['n_v1'] > 0]
a, b = arms(fokc)
obs, p = perm_p(a, b)
print(f"first tranche, clean rule (t={len(a)}, c={len(b)}): "
      f"{a.mean():.3f} vs {b.mean():.3f}  diff {obs:+.3f}  p={p:.2f}")
f272 = [r for r in [json.loads(l) for l in open('pairs.jsonl')][:272]
        if r['status'] == 'ok' and r['n_v1'] > 0]
from scipy.stats import pearsonr as _pr
for nm_, rows_ in (('first 112 attempts', fok), ('first 272 attempts', f272), ('full corpus', ok)):
    x_ = np.log([r['n_v1'] for r in rows_])
    y_ = [r['dropped_share'] for r in rows_]
    r_, pv_ = _pr(x_, y_)
    print(f"log-size vs churn, {nm_}: r = {r_:+.2f} (p={pv_:.2f}, n={len(rows_)})")

print("\n== Section 5: out-of-sample cells ==")
for cell in ('cv25', 'lgcl24', 'cv25b'):
    path = f'pairs_{cell}.jsonl'
    if not os.path.exists(path):
        continue
    rows = [json.loads(l) for l in open(path)]
    cok = [r for r in rows if r['status'] == 'ok' and r['n_v1'] > 0]
    d = np.array([r['dropped_share'] for r in cok])
    a = np.array([r['dropped_share'] for r in cok if r['arm'] == 'treatment'])
    b = np.array([r['dropped_share'] for r in cok if r['arm'] == 'control'])
    obs, pv = perm_p(a, b)
    tot = sum(r['n_v1'] for r in cok); drp = sum(r['dropped'] for r in cok)
    comb_ = np.concatenate([a, b])
    rk = np.empty(len(comb_), float)
    rk[comb_.argsort()] = np.arange(1, len(comb_) + 1)
    for v_ in np.unique(comb_):
        m_ = comb_ == v_
        rk[m_] = rk[m_].mean()
    uo = rk[:len(a)].mean() - rk[len(a):].mean()
    rC = np.random.default_rng(0)
    nullr = np.array([(lambda q: q[:len(a)].mean() - q[len(a):].mean())(rC.permutation(rk))
                      for _ in range(20000)])
    print(f"{cell}: usable {len(cok)} (t={len(a)}, c={len(b)})  "
          f"zero={(d==0).mean():.0%} 1-20%={((d>0)&(d<=0.2)).mean():.0%} "
          f">20%={(d>0.2).mean():.0%} >half={(d>0.5).mean():.0%} total={(d==1).mean():.0%}  "
          f"pooled={drp/tot:.1%} of {tot:,}  "
          f"arm diff {obs:+.3f} p={pv:.2f} rank p={(np.abs(nullr) >= abs(uo)).mean():.2f}  "
          f"treatment-zero={(a==0).mean():.0%}")
    if cell == 'cv25b':
        rD = np.random.default_rng(0)
        bt_ = np.array([rD.choice(a, len(a), True).mean() - rD.choice(b, len(b), True).mean()
                        for _ in range(20000)])
        se_ = np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
        print(f"  cv25b bootstrap [{np.percentile(bt_, 2.5):+.2f}, {np.percentile(bt_, 97.5):+.2f}], "
              f"80% MDE {se_*(1.96+0.84):.2f}")

ids, pool = set(), []
for cell in ('cv25', 'cv25b'):
    path = f'pairs_{cell}.jsonl'
    if not os.path.exists(path):
        continue
    for l in open(path):
        r = json.loads(l)
        if r['status'] == 'ok' and r.get('n_v1', 0) > 0 and r['id'] not in ids:
            ids.add(r['id']); pool.append(r)
if pool:
    a = np.array([r['dropped_share'] for r in pool if r['arm'] == 'treatment'])
    b = np.array([r['dropped_share'] for r in pool if r['arm'] == 'control'])
    obs, pv = perm_p(a, b)
    print(f"pooled cs.CV (pre-registered): t={len(a)} c={len(b)}  "
          f"{a.mean():.3f} vs {b.mean():.3f}  diff {obs:+.3f}  p={pv:.3f}")

fail_t = sum(1 for r in raw if r['arm'] == 'treatment' and r['status'] != 'ok')
fail_c = sum(1 for r in raw if r['arm'] == 'control' and r['status'] != 'ok')
n_t = sum(1 for r in raw if r['arm'] == 'treatment')
from scipy import stats as _st
fp = _st.fisher_exact([[fail_t, n_t-fail_t], [fail_c, len(raw)-n_t-fail_c]])[1]
print(f"retrievability by arm: {fail_t}/{n_t} vs {fail_c}/{len(raw)-n_t} fail, Fisher p={fp:.3f}")
ests = []
cellmap = {'main': ok}
for c_ in ('cv25', 'lgcl24', 'cv25b'):
    if os.path.exists(f'pairs_{c_}.jsonl'):
        rows_ = [json.loads(l) for l in open(f'pairs_{c_}.jsonl')]
        cellmap[c_] = [r for r in rows_ if r['status'] == 'ok' and r['n_v1'] > 0]
for name, rows_ in cellmap.items():
    a_ = np.array([r['dropped_share'] for r in rows_ if r['arm'] == 'treatment'])
    b_ = np.array([r['dropped_share'] for r in rows_ if r['arm'] == 'control'])
    ests.append((a_.mean()-b_.mean(), np.sqrt(a_.var(ddof=1)/len(a_)+b_.var(ddof=1)/len(b_))))
w_ = [1/s**2 for _, s in ests]
db = sum(wi*d for wi, (d, _) in zip(w_, ests))/sum(w_)
Q = sum(wi*(d-db)**2 for wi, (d, _) in zip(w_, ests))
print(f"Cochran Q across 4 arm estimates: {Q:.1f} (normal-theory df=3 p={1-_st.chi2.cdf(Q,3):.3f})")
def _qstat(cellsets):
    ests_ = []
    for rows_ in cellsets:
        tq = np.array([v for v, a_ in rows_ if a_ == 'treatment'])
        cq = np.array([v for v, a_ in rows_ if a_ == 'control'])
        dq = tq.mean() - cq.mean()
        sq = np.sqrt(tq.var(ddof=1)/len(tq) + cq.var(ddof=1)/len(cq))
        ests_.append((dq, sq))
    wq = [1/s**2 for _, s in ests_]
    dbq = sum(wi*d for wi, (d, _) in zip(wq, ests_)) / sum(wq)
    return sum(wi*(d-dbq)**2 for wi, (d, _) in zip(wq, ests_))
baseQ = [[(r['dropped_share'], r['arm']) for r in rows_] for rows_ in cellmap.values()]
rngQ = np.random.default_rng(0)
hits = 0
# 20,000 reps, matching perm_p's default, so every permutation test in the
# ledger is calibrated at the same replicate count.
QREPS = 20000
for _ in range(QREPS):
    shufQ = []
    for rows_ in baseQ:
        armsQ = np.array([a_ for _, a_ in rows_])
        valsQ = [v for v, _ in rows_]
        shufQ.append(list(zip(valsQ, rngQ.permutation(armsQ))))
    if _qstat(shufQ) >= Q:
        hits += 1
print(f"  permutation-calibrated Q ({QREPS:,} reps, arm labels shuffled within cells): "
      f"p = {hits/QREPS:.3f}")
from collections import defaultdict as _dd
strataM = _dd(list)
for r in ok:
    strataM[r['id'][:4]].append(r)
tM = np.array([r['dropped_share'] for r in ok if r['arm'] == 'treatment'])
cM = np.array([r['dropped_share'] for r in ok if r['arm'] == 'control'])
obsM = tM.mean() - cM.mean()
rngM = np.random.default_rng(0)
nullM = []
for _ in range(20000):
    tv_, cv_ = [], []
    for rows_ in strataM.values():
        armsM = np.array([r['arm'] for r in rows_])
        valsM = [r['dropped_share'] for r in rows_]
        permM = rngM.permutation(armsM)
        tv_ += [v for v, a_ in zip(valsM, permM) if a_ == 'treatment']
        cv_ += [v for v, a_ in zip(valsM, permM) if a_ == 'control']
    nullM.append(np.mean(tv_) - np.mean(cv_))
nullM = np.array(nullM)
print(f"month-stratified arm diff ({len(strataM)} strata): {obsM:+.3f}, "
      f"p = {(np.abs(nullM) >= abs(obsM)).mean():.2f}")
strataV = _dd(list)
for r in ok:
    strataV[min(r['v_last'], 5)].append(r)
rngV = np.random.default_rng(0)
nullV = []
for _ in range(20000):
    tv_, cv_ = [], []
    for rows_ in strataV.values():
        armsV = np.array([r['arm'] for r in rows_])
        valsV = [r['dropped_share'] for r in rows_]
        permV = rngV.permutation(armsV)
        tv_ += [v for v, a_ in zip(valsV, permV) if a_ == 'treatment']
        cv_ += [v for v, a_ in zip(valsV, permV) if a_ == 'control']
    nullV.append(np.mean(tv_) - np.mean(cv_))
nullV = np.array(nullV)
print(f"version-stratified arm diff ({len(strataV)} strata): {obsM:+.3f}, "
      f"p = {(np.abs(nullV) >= abs(obsM)).mean():.2f}")
print(f"v1-number medians by arm: treatment "
      f"{np.median([r['n_v1'] for r in ok if r['arm'] == 'treatment']):.0f}, "
      f"control {np.median([r['n_v1'] for r in ok if r['arm'] == 'control']):.0f}")
def _q(items):
    w2 = [1/s**2 for _, s in items]
    db2 = sum(wi*d for wi, (d, _) in zip(w2, items)) / sum(w2)
    return sum(wi*(d-db2)**2 for wi, (d, _) in zip(w2, items))
for i_, nm_ in enumerate(cellmap):
    print(f"  Q without {nm_}: {_q([e for j_, e in enumerate(ests) if j_ != i_]):.1f}")
ACC = re.compile(r'accept|camera[- ]ready|to appear', re.I)
nine = sum(1 for r in ok if r['arm'] == 'control' and ACC.search(r.get('comment', '')))
print(f"controls declaring acceptance in wordings the arm regex missed: {nine} of "
      f"{sum(1 for r in ok if r['arm'] == 'control')}")
att_t = sum(1 for r in raw if r['arm'] == 'treatment'
            and not (r['status'] == 'ok' and r.get('n_v1', 0) > 0))
att_c = sum(1 for r in raw if r['arm'] == 'control'
            and not (r['status'] == 'ok' and r.get('n_v1', 0) > 0))
fp2 = _st.fisher_exact([[att_t, n_t-att_t], [att_c, len(raw)-n_t-att_c]])[1]
t_, c_ = arms(ok)
am_ = (t_.sum() + t_.mean()*att_t)/(len(t_)+att_t) - (c_.sum() + c_.mean()*att_c)/(len(c_)+att_c)
lo_ = t_.sum()/(len(t_)+att_t) - (c_.sum()+att_c)/(len(c_)+att_c)
hi_ = (t_.sum()+att_t)/(len(t_)+att_t) - c_.sum()/(len(c_)+att_c)
print(f"total attrition {att_t}/{n_t} vs {att_c}/{len(raw)-n_t} (Fisher p={fp2:.3f}); "
      f"arm-mean imputation {am_:+.3f}; worst-case [{lo_:+.3f}, {hi_:+.3f}]")
print(f"multiplicity: 29-test ledger; P(>=4 nominal hits | 29, alpha .05) = "
      f"{1 - _st.binom.cdf(3, 29, 0.05):.3f} under (false) independence; "
      f"the four hits share the same cs.CV papers")
if os.path.exists('orv_probe.json'):
    orv = json.load(open('orv_probe.json'))
    matched = [r for r in orv if r['or_match']]
    conf = sum(1 for r in matched if 'CoRR' not in (r.get('or_venue') or ''))
    mirr = len(matched) - conf
    print(f"OpenReview arm-label probe: {conf}/{len(orv)} venue-confirmed, "
          f"{mirr} preprint-mirror only, {len(orv) - len(matched)} no match, "
          f"0 contradicted")
if os.path.exists('pairs_v1_buggy.jsonl'):
    oldb = {r['id']: r for r in map(json.loads, open('pairs_v1_buggy.jsonl'))
            if r['status'] == 'ok' and r.get('n_v1', 0) > 0}
    shb = [p for p in ok if p['id'] in oldb]
    evo = sum(oldb[p['id']]['n_v1'] for p in shb)
    evn = sum(p['n_v1'] for p in shb)
    print(f"extractor bug excess match events: {(evo-evn)/evo:.1%} "
          f"({evo:,} buggy vs {evn:,} fixed on {len(shb)} shared papers)")
print("version-count strata (zero-share / >20% share):")
for lab, f_ in (('v=2', lambda v: v == 2), ('v>=3', lambda v: v >= 3)):
    g_ = [p for p in ok if f_(p['v_last'])]
    dg = np.array([p['dropped_share'] for p in g_])
    print(f"  {lab}: n={len(g_)} zero={(dg == 0).mean():.1%} >20%={(dg > 0.2).mean():.1%}")
mo_ = [int(p['id'][2:4]) for p in ok]
vv_ = [p['v_last'] for p in ok]
rho_, pvr = _st.spearmanr(mo_, vv_)
czt = np.array([float(r['dropped'] == 0) for r in ok if r['arm'] == 'treatment'])
czc = np.array([float(r['dropped'] == 0) for r in ok if r['arm'] == 'control'])
ozs, pzs = perm_p(czt, czc)
print(f"zero-share arm test: {czt.mean():.1%} vs {czc.mean():.1%}, diff {ozs:+.3f}, p={pzs:.2f}")
if os.path.exists('anchored_full.jsonl'):
    afm = {r['id']: r for r in map(json.loads, open('anchored_full.jsonl'))
           if r['status'] == 'ok' and r['anchored_total'] and r['anchored_total'] >= 5}
    ta_ = np.array([afm[r['id']]['anchored_share'] for r in ok
                    if r['arm'] == 'treatment' and r['id'] in afm])
    ca_ = np.array([afm[r['id']]['anchored_share'] for r in ok
                    if r['arm'] == 'control' and r['id'] in afm])
    oan, pan = perm_p(ta_, ca_)
    print(f"anchored arm test: {ta_.mean():.3f} vs {ca_.mean():.3f}, diff {oan:+.3f}, p={pan:.2f}")
# Halving the true labeled fraction doubles the detectable-effect floor.
print(f"effective MDE at 50% control contamination: {MDE/0.5:.3f}")
print(f"censoring: submission month vs version count Spearman rho={rho_:.2f} p={pvr:.4f}")
if 'cv25' in cellmap and 'cv25b' in cellmap:
    def _ad(rows_):
        aa = np.array([r['dropped_share'] for r in rows_ if r['arm'] == 'treatment'])
        bb = np.array([r['dropped_share'] for r in rows_ if r['arm'] == 'control'])
        return aa.mean()-bb.mean(), aa, bb
    dA, tA, cA = _ad(cellmap['cv25']); dB, tB, cB = _ad(cellmap['cv25b'])
    seAB = np.sqrt(tA.var(ddof=1)/len(tA) + cA.var(ddof=1)/len(cA)
                   + tB.var(ddof=1)/len(tB) + cB.var(ddof=1)/len(cB))
    zAB = (dA-dB)/seAB
    rAB = np.random.default_rng(0)
    bsAB = [(rAB.choice(tA, len(tA), True).mean() - rAB.choice(cA, len(cA), True).mean())
            - (rAB.choice(tB, len(tB), True).mean() - rAB.choice(cB, len(cB), True).mean())
            for _ in range(20000)]
    print(f"cv25 vs cv25b direct contrast: {dA-dB:+.3f}, z={zAB:.2f}, "
          f"p={2*(1-_st.norm.cdf(abs(zAB))):.3f}, "
          f"bootstrap [{np.percentile(bsAB, 2.5):+.2f}, {np.percentile(bsAB, 97.5):+.2f}]")

print("\n== Section 5 addendum: when churn happens (longitudinal) ==")
if os.path.exists('longitudinal.jsonl'):
    lon = [json.loads(l) for l in open('longitudinal.jsonl')]
    lok = [r for r in lon if r['status'] == 'ok']
    print(f"multi-version papers fetched: {len(lon)}, usable {len(lok)}")
    chg = [r for r in lok if r['dropped_final'] > 0]
    fr = np.array([r['dropped_early'] / r['dropped_final'] for r in chg])
    pooled = sum(r['dropped_early'] for r in chg) / sum(r['dropped_final'] for r in chg)
    print(f"papers with any final churn: {len(chg)}; share of final churn already "
          f"present at v2: pooled {pooled:.0%}, median {np.median(fr):.0%}, "
          f"mean {fr.mean():.0%}")
    for arm in ('treatment', 'control'):
        a = [r for r in chg if r['arm'] == arm]
        if a:
            p = sum(r['dropped_early'] for r in a) / sum(r['dropped_final'] for r in a)
            print(f"  {arm}: n={len(a)}, pooled early share {p:.0%}")
    a = np.array([r['dropped_early']/r['dropped_final'] for r in chg if r['arm'] == 'treatment'])
    b = np.array([r['dropped_early']/r['dropped_final'] for r in chg if r['arm'] == 'control'])
    obs, p = perm_p(a, b)
    rT = np.random.default_rng(0)
    bt = np.array([rT.choice(a, len(a), True).mean() - rT.choice(b, len(b), True).mean()
                   for _ in range(20000)])
    print(f"paper-level early-share arm diff: {obs:+.3f}  p={p:.2f}  "
          f"CI [{np.percentile(bt,2.5):+.2f}, {np.percentile(bt,97.5):+.2f}]")
    if os.path.exists('longitudinal_v1_buggy.jsonl'):
        old_ = {r['id']: r for r in map(json.loads, open('longitudinal_v1_buggy.jsonl'))
                if r['status'] == 'ok'}
        def _pool(dct, ids=None):
            rows_ = [r for k, r in dct.items()
                     if (ids is None or k in ids) and r['dropped_final'] > 0]
            return (sum(r['dropped_early'] for r in rows_)
                    / sum(r['dropped_final'] for r in rows_), len(rows_))
        new_ = {r['id']: r for r in lok}
        pb, nb = _pool(old_)
        pm, nm = _pool(new_, set(old_))
        pf, nf = _pool(new_)
        print(f"extractor-bug decomposition: buggy 90 ids {pb:.1%} ({nb} changers) -> "
              f"fixed same ids {pm:.1%} ({nm}); full corpus {pf:.1%} ({nf} changers)")

if os.path.exists('fp_survival.json'):
    print("\n== Section 3: false-positive survival ==")
    fs = json.load(open('fp_survival.json'))
    print(f"labeled false positives still present in final version: "
          f"{round(fs['survival_rate']*fs['checked'])}/{fs['checked']} = {fs['survival_rate']:.1%}")

if os.path.exists('recall_probe.jsonl'):
    print("\n== Section 3: recall probe (number classes the extractor rejects) ==")
    rp = [json.loads(l) for l in open('recall_probe.jsonl')]
    z_all = sum(1 for r in rp if r['status'] == 'ok' and r['result_share'] == 0)
    z_none = sum(1 for r in rp if r['status'] == 'ok' and r['result_share'] == 0
                 and r['n_v1'] == 0)
    print(f"zero-churn papers: {z_all} total, {z_none} with no rejected-class "
          f"numbers (excluded from shares below)")
    rok = [r for r in rp if r['status'] == 'ok' and r['n_v1'] > 0]
    for name, grp in (('zero-droppers', [r for r in rok if r['result_share'] == 0]),
                      ('changers', [r for r in rok if r['result_share'] > 0])):
        d_ = np.array([r['dropped_share'] for r in grp])
        pool_ = sum(r['dropped'] for r in grp) / sum(r['n_v1'] for r in grp)
        print(f"{name}: n={len(grp)} (median {np.median([r['n_v1'] for r in grp]):.0f} "
              f"missed-class numbers) pooled {pool_:.1%} median {np.median(d_):.3f} "
              f"zero {(d_ == 0).mean():.0%}")
    z_ = [r for r in rok if r['result_share'] == 0]
    drz = np.array([r['dropped'] for r in z_]); nvz = np.array([r['n_v1'] for r in z_])
    rR = np.random.default_rng(0); idz = np.arange(len(z_))
    Pz = [drz[p_].sum()/nvz[p_].sum() for p_ in (rR.choice(idz, len(z_), True) for _ in range(20000))]
    print(f"zero-dropper pooled cluster CI [{np.percentile(Pz, 2.5):.1%}, {np.percentile(Pz, 97.5):.1%}]")
    rpm = {r['id']: r for r in rp if r['status'] == 'ok'}
    strict0 = sum(1 for r in ok if r['dropped'] == 0
                  and (r['id'] not in rpm or rpm[r['id']]['n_v1'] == 0
                       or rpm[r['id']]['dropped'] == 0))
    print(f"papers dropping no table number of any class: {strict0}/{len(ok)} "
          f"= {strict0/len(ok):.1%} (11 unverifiable counted clean)")
    rok_m = {r['id']: r for r in rok}
    both_z = [r for r in ok if r['id'] in rok_m]
    rz_ = np.mean([r['dropped'] == 0 for r in both_z])
    iz_ = np.mean([rok_m[r['id']]['dropped'] == 0 for r in both_z])
    bz = sum(1 for r in both_z if r['dropped'] == 0 and rok_m[r['id']]['dropped'] > 0)
    cz = sum(1 for r in both_z if r['dropped'] > 0 and rok_m[r['id']]['dropped'] == 0)
    print(f"in-table control: result zero {rz_:.1%} vs rejected-class zero {iz_:.1%} "
          f"on {len(both_z)} papers; McNemar {bz} vs {cz}, exact p = "
          f"{_st.binomtest(min(bz, cz), bz + cz, 0.5).pvalue:.3f}")
    dm_ = [r for r in both_z if rok_m[r['id']]['n_v1'] >= r['n_v1']]
    rz2 = np.mean([r['dropped'] == 0 for r in dm_])
    iz2 = np.mean([rok_m[r['id']]['dropped'] == 0 for r in dm_])
    b2_ = sum(1 for r in dm_ if r['dropped'] == 0 and rok_m[r['id']]['dropped'] > 0)
    c2_ = sum(1 for r in dm_ if r['dropped'] > 0 and rok_m[r['id']]['dropped'] == 0)
    print(f"  denominator-matched (rejected count >= result count, n={len(dm_)}): "
          f"{rz2:.1%} vs {iz2:.1%}, McNemar p = "
          f"{_st.binomtest(min(b2_, c2_), b2_ + c2_, 0.5).pvalue:.2f}")

if os.path.exists('anchored_full.jsonl'):
    print("\n== Section 3: position-anchored rerun (full corpus) ==")
    af = [json.loads(l) for l in open('anchored_full.jsonl')]
    aok = [r for r in af if r['status'] == 'ok' and r['anchored_total']
           and r['anchored_total'] >= 5]
    adoc = np.array([r['doc_share'] for r in aok])
    aanc = np.array([r['anchored_share'] for r in aok])
    apool = 1 - sum(r['anchored_kept'] for r in aok) / sum(r['anchored_total'] for r in aok)
    print(f"papers with >=5 anchored cells: {len(aok)} "
          f"(median {np.median([r['anchored_total'] for r in aok]):.0f} cells, "
          f"{sum(r['anchored_total'] for r in aok):,} total)")
    print(f"document matcher on this subset: zero {(adoc == 0).mean():.0%} "
          f"mid {((adoc > 0) & (adoc <= 0.2)).mean():.0%} >20% {(adoc > 0.2).mean():.0%}")
    print(f"anchored matcher: zero {(aanc == 0).mean():.0%} "
          f"mid {((aanc > 0) & (aanc <= 0.2)).mean():.1%} >20% {(aanc > 0.2).mean():.0%}  "
          f"pooled {apool:.1%} mean {aanc.mean():.3f} median {np.median(aanc):.3f}")
    rngA = np.random.default_rng(0)
    bzA = [(aanc[rngA.integers(0, len(aanc), len(aanc))] == 0).mean()
           for _ in range(20000)]
    print(f"anchored zero-share paper bootstrap 95% CI: "
          f"[{np.percentile(bzA, 2.5):.0%}, {np.percentile(bzA, 97.5):.0%}]")
    zd = adoc == 0
    print(f"doc-zero papers ({zd.sum()}): anchored-zero {np.mean(aanc[zd] == 0):.0%}, "
          f"<=5% movement {np.mean(aanc[zd] <= 0.05):.0%}, "
          f"median anchored {np.median(aanc[zd]):.3f} vs rest {np.median(aanc[~zd]):.2f}, "
          f"mean {aanc[zd].mean():.2f}, >20% {np.mean(aanc[zd] > 0.2):.0%}")

if os.path.exists('drop_taxonomy.jsonl'):
    print("\n== Section 2: what a dropped number is ==")
    import collections as _cc
    dt = [json.loads(l) for l in open('drop_taxonomy.jsonl')]
    dtok = [r for r in dt if r['status'] == 'ok']
    tot_ = _cc.Counter()
    for r in dtok:
        tot_.update(r['counts'])
    ndrop = sum(tot_.values())
    print(f"papers {len(dtok)}, drops classified {ndrop:,}")
    for k_ in ('formatting', 'value-change', 'row-change', 'gone'):
        print(f"  {k_}: {tot_[k_]:,} ({tot_[k_]/ndrop:.1%})")

if os.path.exists('drop_labels.json'):
    import collections as _cc
    samp_ = json.load(open('drop_labels_sample.json'))
    auth_ = json.load(open('drop_labels.json'))
    blnd_ = json.load(open('drop_labels_blind.json'))
    al_ = {i + 1: r['classifier'] for i, r in enumerate(samp_)}
    for e_ in auth_['errors']:
        al_[e_['index']] = e_['correct']
    bl2_ = {i + 1: r['classifier'] for i, r in enumerate(samp_)}
    for d_ in blnd_['disagree']:
        bl2_[d_['index']] = d_['better']
    cl_ = {i + 1: r['classifier'] for i, r in enumerate(samp_)}
    n_ = len(samp_)
    def _bnd(c): return c == 'formatting'
    a_ag = sum(al_[i] == cl_[i] for i in cl_)
    b_ag = sum(bl2_[i] == cl_[i] for i in cl_)
    b_bnd = sum(_bnd(bl2_[i]) == _bnd(cl_[i]) for i in cl_)
    flags_ = [(i, lab[i]) for lab in (al_, bl2_) for i in cl_
              if _bnd(lab[i]) != _bnd(cl_[i])]
    onedir_ = all(cl_[i] == 'formatting' for i, _ in flags_)
    print(f"classifier validation: n={n_} sampled drops "
          f"({_cc.Counter(cl_.values())})")
    print(f"  author agrees {a_ag}/{n_}; blind LLM agrees {b_ag}/{n_}, "
          f"{b_bnd}/{n_} at formatting boundary")
    print(f"  boundary flags {len(flags_)} over {len(set(i for i,_ in flags_))} rows; "
          f"all formatting over-calls: {onedir_}")

if os.path.exists('baseline.jsonl'):
    print("\n== Section 4: non-result baseline ==")
    bl_ = [json.loads(l) for l in open('baseline.jsonl')]
    bok = [r for r in bl_ if r['status'] == 'ok' and r['n_v1'] > 0]
    d_b = np.array([r['dropped_share'] for r in bok])
    pooled_b = sum(r['dropped'] for r in bok) / sum(r['n_v1'] for r in bok)
    print(f"non-result decimals (outside tables, no %/pm): {len(bok)} papers, "
          f"{sum(r['n_v1'] for r in bok):,} v1 numbers, pooled churn {pooled_b:.1%}, "
          f"mean {d_b.mean():.3f} median {np.median(d_b):.3f} zero {(d_b == 0).mean():.0%}")
    okm = {r['id']: r for r in ok}
    mc_b = sum(1 for r in bok if r['id'] in okm
               and okm[r['id']]['dropped'] == 0 and r['dropped'] > 0)
    mc_c = sum(1 for r in bok if r['id'] in okm
               and okm[r['id']]['dropped'] > 0 and r['dropped'] == 0)
    print(f"McNemar on paired zero indicators: result-zero-only {mc_b}, "
          f"ambient-zero-only {mc_c}, exact p = "
          f"{_st.binomtest(mc_b, mc_b + mc_c, 0.5).pvalue:.1e}")
    dd = np.array([okm[r['id']]['dropped_share'] - r['dropped_share']
                   for r in bok if r['id'] in okm])
    print(f"paired result-minus-nonresult churn: mean {dd.mean():+.3f} "
          f"median {np.median(dd):+.3f}, result side higher in {(dd > 0).mean():.0%} "
          f"of {len(dd)} papers, lower in {(dd < 0).mean():.0%}, "
          f"Wilcoxon p={_st.wilcoxon(dd[dd != 0]).pvalue:.2f}")
    blm = {r['id']: r for r in bok}
    csub = [r for r in okc if r['id'] in blm]
    pc_ = sum(r['dropped'] for r in csub) / sum(r['n_v1'] for r in csub)
    pa_ = (sum(blm[r['id']]['dropped'] for r in csub)
           / sum(blm[r['id']]['n_v1'] for r in csub))
    hv2 = max(csub, key=lambda r: r['n_v1'])
    csub2 = [r for r in csub if r['id'] != hv2['id']]
    pc2_ = sum(r['dropped'] for r in csub2) / sum(r['n_v1'] for r in csub2)
    pa2_ = (sum(blm[r['id']]['dropped'] for r in csub2)
            / sum(blm[r['id']]['n_v1'] for r in csub2))
    print(f"clean rule vs ambient on the same {len(csub)} papers: "
          f"{pc_:.1%} vs {pa_:.1%}; ex-outlier {pc2_:.1%} vs {pa2_:.1%}")
    # 0.139 is the FIRST TRANCHE's pooled broad churn, matching the tranche
    # scope of the ambient baseline this section reports.
    print(f"precision-corrected broad churn (first tranche): "
          f"{(0.139 - 0.18*(1 - 0.946))/0.82:.1%}")
    print(f"controls with empty comment field: "
          f"{sum(1 for r in ok if r['arm'] == 'control' and not r.get('comment', '').strip())}"
          f"/{sum(1 for r in ok if r['arm'] == 'control')}")
    smp1 = json.load(open('precision_sample.json'))
    fp1 = {i for v in json.load(open('precision.json'))['false_positives'].values() for i in v}
    tab1 = [i for i, rec_ in enumerate(smp1, 1) if rec_['kind'] == 'tabular']
    smp2 = json.load(open('precision_sample2.json'))
    fp2_ = {i for v in json.load(open('precision2.json'))['not_result'].values() for i in v}
    tab2 = [i for i, rec_ in enumerate(smp2, 1) if rec_['kind'] == 'tabular']
    a1, b1 = sum(1 for i in tab1 if i not in fp1), sum(1 for i in tab1 if i in fp1)
    a2, b2 = sum(1 for i in tab2 if i not in fp2_), sum(1 for i in tab2 if i in fp2_)
    print(f"tranche tabular precision swing: {a1}/{len(tab1)} vs {a2}/{len(tab2)}, "
          f"Fisher p={_st.fisher_exact([[a1, b1], [a2, b2]])[1]:.3f}")
