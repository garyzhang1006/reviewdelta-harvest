#!/usr/bin/env python3
"""
Stage 5: hand-labeled extraction precision.

One author read all 150 sampled numbers with 120 characters of surrounding
LaTeX and marked each as a reported result or not. NOT_RESULT lists the
indices judged not to be results, grouped by why. Everything else is a result.

Why the direction of this error matters: a non-result that the extractor
accepts is usually a number that does not change between versions, such as a
learning rate or a model name. It therefore lands in "shared", inflating the
denominator of dropped-share without touching the numerator. False positives
push the churn estimate down, so the measured churn is a lower bound.
"""
import json

NOT_RESULT = {
    'hyperparameter': [22, 24, 98, 99, 108, 144, 145, 146, 147],
    'model-name version digit': [39, 43, 44, 73],
    'latex rendering parameter': [15],
    'experimental setting, not an outcome': [18, 19, 58, 60, 105, 109, 110, 111],
    'dataset or model specification': [7, 46, 47, 48],
    'normalization anchor (1.0x)': [14],
}

if __name__ == '__main__':
    sample = json.load(open('precision_sample.json'))
    n = len(sample)
    bad = sorted(i for v in NOT_RESULT.values() for i in v)
    assert len(bad) == len(set(bad)), "an index was labeled twice"
    assert max(bad) <= n, f"label index {max(bad)} exceeds sample size {n}"
    prec = (n - len(bad)) / n
    print(f"hand-labeled {n} sampled numbers from the extractor's output")
    print(f"precision: {n - len(bad)}/{n} = {prec:.1%}\n")
    print("false positives by cause:")
    for k, v in sorted(NOT_RESULT.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(v):2d}  {k}")
    # Do the false positives cluster in one extraction rule?
    kinds = {}
    for i in bad:
        kinds[sample[i - 1]['kind']] = kinds.get(sample[i - 1]['kind'], 0) + 1
    tot = {}
    for s in sample:
        tot[s['kind']] = tot.get(s['kind'], 0) + 1
    print("\nprecision by extraction rule:")
    for k in tot:
        print(f"  {k:8s}: {(tot[k]-kinds.get(k,0))/tot[k]:.1%}  (n={tot[k]})")
    json.dump({'n': n, 'precision': prec, 'false_positives': NOT_RESULT},
              open('precision.json', 'w'), indent=1)
