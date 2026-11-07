"""linearity_v2.py - held-out linearity of the user-view -> assistant-view embedding map (v2, numpy only).

Input: .npz files written by dump_embeddings.py (arrays u, a: N x d, one row per test pair).
Per file: 5-fold cross-validation over pairs. Inside each training fold both sides are centred, scaled by ONE global
scalar (so a collapsed model is not penalised by the absolute ridge grid), reduced to 50 principal components, and a
ridge map user -> assistant is fitted (alpha by leave-one-out on the training fold). The prediction is projected back
to the full space. R2 = 1 - sum squared error / sum squared deviation from the mean assistant embedding, over all
held-out pairs. Null: the same with the assistant rows permuted across pairs. Excess = R2 - null R2.
Paired bootstrap over pairs for the difference B - A (averaged over seeds inside each resample).
usage: python linearity_v2.py --base emb --a turk_sft_s{seed} --b turk_static_k0_lam2_s{seed} --seeds 82,23,37
"""
import argparse
import os
import numpy as np

ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
NCOMP = 50
FOLDS = 5


def ridge_loo_alpha(X, Y):
    n, p = X.shape
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    UtY = U.T @ Y
    best, best_err = ALPHAS[0], np.inf
    for al in ALPHAS:
        d = S ** 2 / (S ** 2 + al)
        hii = ((U ** 2) * d).sum(1)
        fit = U @ (d[:, None] * UtY)
        loo = (Y - fit) / (1.0 - hii)[:, None]
        err = float((loo ** 2).mean())
        if err < best_err:
            best, best_err = al, err
    return best


def fit_predict(utr, atr, ute, ncomp=NCOMP):
    mu_u, mu_a = utr.mean(0), atr.mean(0)
    uc, ac = utr - mu_u, atr - mu_a
    su = np.sqrt((uc ** 2).sum(1).mean()) + 1e-12
    sa = np.sqrt((ac ** 2).sum(1).mean()) + 1e-12
    uc, ac, ute_c = uc / su, ac / sa, (ute - mu_u) / su
    k = min(ncomp, uc.shape[0] - 1, uc.shape[1])
    _, _, Vu = np.linalg.svd(uc, full_matrices=False)
    _, _, Va = np.linalg.svd(ac, full_matrices=False)
    Vu, Va = Vu[:k], Va[:k]
    X, Y = uc @ Vu.T, ac @ Va.T
    al = ridge_loo_alpha(X, Y)
    W = np.linalg.solve(X.T @ X + al * np.eye(k), X.T @ Y)
    pred = ((ute_c @ Vu.T) @ W) @ Va  # back to the full normalised space
    return pred * sa + mu_a


def oof_errors(u, a, seed=1234, perm_seed=4321):
    n = u.shape[0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.array_split(order, FOLDS)
    perm = np.random.default_rng(perm_seed).permutation(n)
    a_sh = a[perm]
    e = np.zeros(n)
    es = np.zeros(n)
    for f in folds:
        tr = np.setdiff1d(order, f)
        p = fit_predict(u[tr], a[tr], u[f])
        e[f] = ((a[f] - p) ** 2).sum(1)
        ps = fit_predict(u[tr], a_sh[tr], u[f])
        es[f] = ((a_sh[f] - ps) ** 2).sum(1)
    s = ((a - a.mean(0)) ** 2).sum(1)
    s_sh = ((a_sh - a_sh.mean(0)) ** 2).sum(1)
    return e, es, s, s_sh


def stats(e, es, s, s_sh, idx=None):
    if idx is not None:
        e, es, s, s_sh = e[idx], es[idx], s[idx], s_sh[idx]
    r2 = 1 - e.mean() / s.mean()
    r2n = 1 - es.mean() / s_sh.mean()
    return r2, r2n, r2 - r2n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="emb")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--seeds", default="82,23,37")
    ap.add_argument("--resamples", type=int, default=10000)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    res = {"A": [], "B": []}
    for side, pat in (("A", args.a), ("B", args.b)):
        for sd in seeds:
            z = np.load(os.path.join(args.base, pat.format(seed=sd) + ".npz"))
            u, a = z["u"].astype(np.float64), z["a"].astype(np.float64)
            e, es, s, s_sh = oof_errors(u, a)
            r2, r2n, ex = stats(e, es, s, s_sh)
            cos = float(((u / np.linalg.norm(u, axis=1, keepdims=True)) * (a / np.linalg.norm(a, axis=1, keepdims=True))).sum(1).mean())
            print(f"{side} seed {sd}: n={len(u)} R2={r2:+.4f} null R2={r2n:+.4f} excess={ex:+.4f} | "
                  f"var(u)={((u - u.mean(0)) ** 2).sum(1).mean():.3e} var(a)={s.mean():.3e} mean cos(u,a)={cos:.4f}")
            res[side].append((e, es, s, s_sh))
    n = len(res["A"][0][0])
    rng = np.random.default_rng(1234)
    diffs = []
    done = 0
    while done < args.resamples:
        m = min(1000, args.resamples - done)
        idx = rng.integers(0, n, size=(m, n))
        d = np.zeros(m)
        for ra, rb in zip(res["A"], res["B"]):
            for j in range(m):
                d[j] += stats(*rb, idx[j])[2] - stats(*ra, idx[j])[2]
        diffs.append(d / len(seeds))
        done += m
    diffs = np.concatenate(diffs)
    point = np.mean([stats(*rb)[2] - stats(*ra)[2] for ra, rb in zip(res["A"], res["B"])])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\nexcess linearity, B minus A, mean over seeds: {point:+.4f}; paired bootstrap ({args.resamples}, rng 1234) "
          f"95% interval [{lo:+.4f}, {hi:+.4f}]")
    print(f"H1 criterion for this dataset (interval above 0): {lo > 0}")


if __name__ == "__main__":
    main()
