"""summarize.py - one table for many runs.   usage: python v2\\summarize.py v2\\runs\\pilot_sft_*"""
import glob
import json
import os
import sys

dirs = []
for pat in sys.argv[1:]:
    dirs += sorted(glob.glob(pat))
rows = []
for d in dirs:
    if not os.path.isdir(d):
        continue
    def load(name):
        p = os.path.join(d, name)
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
    a, f = load("args.json"), load("final.json")
    ev = None
    for cand in ("val_eval.json", "test_eval.json"):
        ev = ev or load(cand)
    s = ev["summary"] if ev else {}
    last = f["last"] if f else {}
    rows.append([os.path.basename(d), a.get("arm") if a else "?", a.get("lam") if a else "?", a.get("k") if a else "?",
                 a.get("lr") if a else "?", a.get("seed") if a else "?",
                 round(last.get("lm_loss", float("nan")), 3) if last else "-",
                 round(f["seconds"] / 60, 1) if f else "-",
                 round(s["exec_acc_all"] * 100, 2) if s else "-",
                 round(s["exec_acc_nonempty"] * 100, 2) if s else "-",
                 s.get("pred_errors", "-")])
head = ["run", "arm", "lam", "k", "lr", "seed", "last_lm_loss", "train_min", "exec_all%", "exec_nonempty%", "pred_err"]
w = [max(len(str(x)) for x in [h] + [r[i] for r in rows]) for i, h in enumerate(head)]
print("  ".join(h.ljust(w[i]) for i, h in enumerate(head)))
for r in rows:
    print("  ".join(str(x).ljust(w[i]) for i, x in enumerate(r)))
