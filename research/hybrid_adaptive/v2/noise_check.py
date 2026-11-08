"""noise_check.py - bf16 reproducibility of the test embeddings (amendment 13).
Re-embeds the test pairs of one adapter with batch size 4 and reversed row order, restores the original order,
saves emb/<label>_bs4rev.npz and prints the noise share of the variance against emb/<label>.npz (batch size 16).
usage: python noise_check.py --val FILE --k K --label LABEL --adapter PATH [--emb emb]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import core
from probe_alignment import load, embed_all


def share(x, y):
    noise = ((x - y) ** 2).sum(1).mean() / 2.0
    total = ((x - x.mean(0, keepdims=True)) ** 2).sum(1).mean()
    return noise, total, noise / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--emb", default="emb")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = core.read_jsonl(args.val)
    ref = np.load(os.path.join(args.emb, args.label + ".npz"))
    model, tok = load(args.adapter, dev)
    u, a = embed_all(model, tok, rows[::-1], args.k, dev, bs=4)
    u = u.numpy().astype(np.float32)[::-1].copy()
    a = a.numpy().astype(np.float32)[::-1].copy()
    out = os.path.join(args.emb, args.label + "_bs4rev.npz")
    np.savez(out, u=u, a=a)
    print("saved", out, u.shape)
    for name, x, y in (("user", ref["u"], u), ("assistant", ref["a"], a)):
        n, t, r = share(x.astype(np.float64), y.astype(np.float64))
        print("%s: noise variance %.4g  total variance %.4g  noise share %.4g" % (name, n, t, r))


if __name__ == "__main__":
    main()