"""dump_embeddings.py - save user-view / assistant-view embeddings of the test pairs for linearity_v2.py.
Uses the same loading and embedding code as probe_alignment.py (core.encode, offset -3, max_length 512, bf16).
usage: python dump_embeddings.py --val FILE --k 0 --outdir emb --adapters label=PATH [label=PATH ...]   (PATH 'none' = base)
"""
import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import core
from probe_alignment import load, embed_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--outdir", default="emb")
    ap.add_argument("--adapters", nargs="+", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = core.read_jsonl(args.val)
    os.makedirs(args.outdir, exist_ok=True)
    for item in args.adapters:
        label, path = item.split("=", 1)
        out = os.path.join(args.outdir, label + ".npz")
        if os.path.exists(out):
            print("skip", out)
            continue
        model, tok = load(path, dev)
        u, a = embed_all(model, tok, rows, args.k, dev)
        np.savez(out, u=u.numpy().astype(np.float32), a=a.numpy().astype(np.float32))
        print("saved", out, tuple(u.shape))
        del model
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
