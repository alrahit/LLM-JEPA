"""traj.py - training trajectory of the JEPA diagnostics.   usage: python v2\\traj.py RUN_DIR [every_nth_logged_point]"""
import json
import os
import sys

d = sys.argv[1]
every = int(sys.argv[2]) if len(sys.argv) > 2 else 4
rows = [json.loads(l) for l in open(os.path.join(d, "train_log.jsonl"), encoding="utf-8") if l.strip()]
keys = ["step", "lm_loss", "jepa_loss", "cos_pos", "cos_neg", "retrieval_acc"]
print(f"\n== {os.path.basename(d.rstrip(chr(92) + '/'))}")
print("  ".join(k.ljust(13) for k in keys))
picked = rows[::every]
if rows and rows[-1] is not picked[-1]:
    picked.append(rows[-1])
for r in picked:
    print("  ".join((f"{r[k]:.4f}" if isinstance(r.get(k), float) else str(r.get(k, "-"))).ljust(13) for k in keys))
