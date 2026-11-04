"""eval_exact.py - greedy generation + exact-match accuracy (Synth, Turk and other string-answer tasks).

  --predict model : load base model (+ optional LoRA adapter) and generate (same code as eval_spider.py)
  --predict gold  : use the gold answer as the prediction (scorer sanity check, no GPU needed)

The output JSON has the same layout as eval_spider.py, so compare.py works on it unchanged.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_spider


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", choices=["model", "gold"], default="model")
    ap.add_argument("--model", default="Qwen/Qwen2-1.5B-Instruct")
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_examples", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    args = ap.parse_args()

    rows = eval_spider.read_jsonl(args.data)
    if args.max_examples:
        rows = rows[: args.max_examples]
    golds = [r["messages"][2]["content"] for r in rows]
    preds = golds if args.predict == "gold" else eval_spider.generate_all(args, rows)

    items = []
    for i, (g, p) in enumerate(zip(golds, preds)):
        items.append({"idx": i, "db_id": "", "gold": g, "pred": p, "strict": str(p).strip() == str(g).strip()})
    n = len(items)
    correct = sum(1 for x in items if x["strict"])
    summary = {"n": n, "correct": correct, "exact_match": correct / n if n else 0.0}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "summary": summary, "examples": items}, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
