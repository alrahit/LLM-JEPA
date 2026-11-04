"""Re-scores saved round-1 Spider generations with the v2 execution scorer.
Usage: python rescore_old_spider.py <spider_db_dir> <eval1.json> [<eval2.json> ...]"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spider_exec

db_dir, files = os.path.abspath(sys.argv[1]), sys.argv[2:]
rows_by_file = {}
for p in files:
    details = json.load(open(p, encoding="utf-8"))["details"]
    rows_by_file[p] = details
    gold_items, items = [], []
    for x in details:
        db_id = spider_exec.extract_db_id(x["prompt"])
        gold_items.append(spider_exec.score(db_dir, db_id, x["ground_truth"], x["ground_truth"]))
        items.append(spider_exec.score(db_dir, db_id, x["ground_truth"], x["generated_text"].strip()))
    sha = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    show = lambda s: json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})
    print(f"\n== {os.path.basename(p)}  sha256 {sha}")
    print("   gold-as-prediction sanity:", show(spider_exec.summarize(gold_items)))
    print("   model generations        :", show(spider_exec.summarize(items)))

names = list(rows_by_file)
print()
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = rows_by_file[names[i]], rows_by_file[names[j]]
        same = sum(1 for x, y in zip(a, b) if x["generated_text"] == y["generated_text"])
        print(f"identical generated_text: {os.path.basename(names[i])} vs {os.path.basename(names[j])}: {same}/{min(len(a), len(b))}")
