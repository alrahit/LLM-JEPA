"""make_splits.py - carve a fixed pilot validation split out of the training file (test file is never touched)."""
import argparse
import hashlib
import json
import os
import random

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=r"D:\LLM_JEPA\datasets\spider_train.jsonl")
ap.add_argument("--out_dir", default=r"D:\LLM_JEPA\hybrid_adaptive\v2\data")
ap.add_argument("--val_frac", type=float, default=0.05)
ap.add_argument("--seed", type=int, default=1234)
a = ap.parse_args()

with open(a.src, encoding="utf-8") as f:
    lines = [l.rstrip("\n").rstrip("\r") for l in f if l.strip()]
idx = list(range(len(lines)))
random.Random(a.seed).shuffle(idx)
n_val = int(round(len(lines) * a.val_frac))
val = set(idx[:n_val])
os.makedirs(a.out_dir, exist_ok=True)
tr_path, va_path = os.path.join(a.out_dir, "pilot_train.jsonl"), os.path.join(a.out_dir, "pilot_val.jsonl")
with open(tr_path, "w", encoding="utf-8", newline="\n") as ft, open(va_path, "w", encoding="utf-8", newline="\n") as fv:
    for i, l in enumerate(lines):
        (fv if i in val else ft).write(l + "\n")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
info = {"src": a.src, "seed": a.seed, "val_frac": a.val_frac, "n_total": len(lines), "n_train": len(lines) - n_val,
        "n_val": n_val, "sha_train": sha(tr_path), "sha_val": sha(va_path)}
json.dump(info, open(os.path.join(a.out_dir, "split_info.json"), "w"), indent=1)
print(json.dumps(info, indent=1))
