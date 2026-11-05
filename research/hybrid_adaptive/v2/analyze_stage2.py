import argparse, json, math, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=os.path.join("runs", "stage2"))
ap.add_argument("--a", default="qwen2_synth_sft_lr2e-4_s{seed}")
ap.add_argument("--b", default="qwen2_synth_static_k1_lam0.5_lr2e-4_s{seed}")
ap.add_argument("--seeds", default="82,23,37")
ap.add_argument("--file", default="test_eval.json")
ap.add_argument("--resamples", type=int, default=10000)
ap.add_argument("--rng", type=int, default=1234)
args = ap.parse_args()
seeds = [int(s) for s in args.seeds.split(",")]

def load(name):
    path = os.path.join(args.base, name, args.file)
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    ex = d["examples"]
    return np.array([1 if e["strict"] else 0 for e in ex], dtype=np.int8), [e["gold"] for e in ex], d["summary"]

A, B, gold = [], [], None
for s in seeds:
    for lst, pat in ((A, args.a), (B, args.b)):
        flags, g, summ = load(pat.format(seed=s))
        assert summ["n"] == len(flags), "n mismatch"
        if gold is None:
            gold = g
        assert g == gold, "gold answers differ between runs (different test file or order)"
        lst.append(flags)
A = np.stack(A); B = np.stack(B)
n = A.shape[1]
print(f"examples n = {n}, seeds = {seeds}")
print("seed   LM-only    JEPA     diff (points)   correct A/B")
for i, s in enumerate(seeds):
    a, b = A[i].mean() * 100, B[i].mean() * 100
    print(f"{s:>4}  {a:7.2f}  {b:7.2f}  {b - a:+8.2f}         {int(A[i].sum())}/{int(B[i].sum())}")
a_m, b_m = A.mean() * 100, B.mean() * 100
print(f"mean  {a_m:7.2f}  {b_m:7.2f}  {b_m - a_m:+8.2f}")

D = (B.astype(np.float64) - A.astype(np.float64)).mean(axis=0)
obs = D.mean() * 100
rng = np.random.default_rng(args.rng)
boots = []
left = args.resamples
while left > 0:
    m = min(1000, left)
    idx = rng.integers(0, n, size=(m, n))
    boots.append(D[idx].mean(axis=1) * 100)
    left -= m
boots = np.concatenate(boots)
lo, hi = np.percentile(boots, [2.5, 97.5])
print(f"\nPaired bootstrap ({args.resamples} resamples, rng {args.rng}): mean difference {obs:+.3f} points, 95% interval [{lo:+.3f}, {hi:+.3f}]")

b_only = int(((A == 0) & (B == 1)).sum())
c_only = int(((A == 1) & (B == 0)).sum())
tot = b_only + c_only
if tot:
    k = min(b_only, c_only)
    p = min(1.0, 2 * sum(math.comb(tot, i) for i in range(k + 1)) / 2 ** tot)
else:
    p = 1.0
print(f"Exact McNemar on pooled predictions (descriptive): JEPA-only correct {b_only}, LM-only correct {c_only}, p = {p:.4f}")

print("\nPre-registered H0 criteria for this dataset (Holm across Synth and Turk is applied when both exist):")
print(f"  mean difference at least 1.0 point: {obs >= 1.0}")
print(f"  95% interval above 0:               {lo > 0}")
