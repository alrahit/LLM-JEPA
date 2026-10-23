"""eval_spider.py - greedy generation + execution accuracy on Spider.

  --predict model : load base model (+ optional LoRA adapter) and generate
  --predict gold  : use the gold SQL as the prediction (sanity check of the scorer, no GPU needed)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spider_exec


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def generate_all(args, rows):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    if args.adapter and args.adapter.lower() != "none":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    prompts = [tok.apply_chat_template(r["messages"][:2], tokenize=False, add_generation_prompt=True) for r in rows]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    preds = [None] * len(rows)
    t0 = time.time()
    for s in range(0, len(order), args.batch_size):
        idx = order[s: s + args.batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 temperature=None, top_p=None, top_k=None,
                                 repetition_penalty=1.0,  # Qwen2-Instruct's generation_config has 1.1: override it
                                 eos_token_id=eos_ids, pad_token_id=tok.pad_token_id, use_cache=True)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, t in zip(idx, texts):
            preds[i] = t.strip()
        print(f"generated {min(s + args.batch_size, len(order))}/{len(order)}  {time.time() - t0:.0f}s", flush=True)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", choices=["model", "gold"], default="model")
    ap.add_argument("--model", default="Qwen/Qwen2-1.5B-Instruct")
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--data", required=True)
    ap.add_argument("--db_dir", default=r"D:\LLM_JEPA\datasets\spider_db")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_examples", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=192)
    args = ap.parse_args()

    rows = read_jsonl(args.data)
    if args.max_examples:
        rows = rows[: args.max_examples]
    golds = [r["messages"][2]["content"] for r in rows]
    preds = golds if args.predict == "gold" else generate_all(args, rows)

    items = []
    for i, (r, g, p) in enumerate(zip(rows, golds, preds)):
        db_id = spider_exec.extract_db_id(r["messages"][1]["content"])
        sc = spider_exec.score(args.db_dir, db_id, g, p)
        sc.update({"idx": i, "db_id": db_id, "gold": g, "pred": p})
        items.append(sc)
    summary = spider_exec.summarize(items)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "summary": summary, "examples": items}, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
