"""logstats.py - per-epoch averages from train_log.jsonl.   usage: python v2\\logstats.py RUN_DIR [RUN_DIR ...]"""
import collections
import json
import os
import sys

KEYS = ["lm_loss", "grad_norm", "jepa_loss", "cos_pos", "cos_neg", "retrieval_acc"]
for d in sys.argv[1:]:
    p = os.path.join(d, "train_log.jsonl")
    if not os.path.exists(p):
        print("missing", p)
        continue
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["epoch"]].append(r)
    print(f"\n== {os.path.basename(d.rstrip(chr(92) + '/'))}  ({len(rows)} logged points)")
    print("epoch  " + "  ".join(k.ljust(13) for k in KEYS))
    for ep in sorted(by):
        cells = []
        for k in KEYS:
            vals = [r[k] for r in by[ep] if k in r and r[k] == r[k]]
            cells.append((f"{sum(vals) / len(vals):.4f}" if vals else "-").ljust(13))
        print(f"{ep:<5}  " + "  ".join(cells))
