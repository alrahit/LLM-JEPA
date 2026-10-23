"""probe_alignment.py - offline alignment probe on the validation pairs (text view vs SQL view).

For every model it embeds the SAME (user view, assistant view) pairs and reports, over all N pairs:
  cos_pos / cos_neg   mean cosine of true pairs / of all mismatched pairs   (raw)
  *_c                 same after subtracting each side's mean embedding      (removes the shared direction)
  r1 / r5             retrieval@1 / @5 among all N candidates               (chance = 1/N, 5/N)
  aniso_u             mean cosine between any two user-view embeddings       (1.0 = everything points one way)
Two settings: k=3 (user view with 3 predictor tokens, as in static training) and k=0 (no predictor tokens).
usage: python v2\\probe_alignment.py --val v2\\data\\pilot_val.jsonl --adapters base=none sft=PATH static=PATH
"""
import argparse
import gc
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F

import core

MODEL = "Qwen/Qwen2-1.5B-Instruct"
OFFSET = -3


def load(adapter, dev):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": [f"<|predictor_{i}|>" for i in range(1, 4)]})
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    core.init_predictor_embeddings(base, tok, 3)  # same deterministic init as in training
    base.to(dev)
    model = base if adapter == "none" else PeftModel.from_pretrained(base, adapter)
    model.eval()
    return model, tok


@torch.no_grad()
def embed_all(model, tok, rows, k, dev, bs=16):
    exs = [core.build_example(tok, r["messages"], k, 512, True) for r in rows]
    us, as_ = [], []
    for s in range(0, len(exs), bs):
        b = core.collate(exs[s: s + bs], tok.pad_token_id)
        b = {key: v.to(dev) for key, v in b.items()}
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
            hu = core.encode(model, b["user_ids"], b["user_mask"])
            ha = core.encode(model, b["asst_ids"], b["asst_mask"])
            us.append(core.last_token_embedding(hu, b["user_mask"], OFFSET).float().cpu())
            as_.append(core.last_token_embedding(ha, b["asst_mask"], OFFSET).float().cpu())
    return torch.cat(us), torch.cat(as_)


def block(x, y, tag, out):
    n = x.size(0)
    sim = F.normalize(x, dim=-1) @ F.normalize(y, dim=-1).T
    diag = sim.diag()
    out["cos_pos" + tag] = diag.mean().item()
    out["cos_neg" + tag] = ((sim.sum() - diag.sum()) / (n * (n - 1))).item()
    better = (sim > diag.unsqueeze(1)).sum(dim=1)  # candidates strictly closer than the true partner
    out["r1" + tag] = (better == 0).float().mean().item()
    out["r5" + tag] = (better < 5).float().mean().item()


def metrics(u, a):
    out = {}
    block(u, a, "", out)
    block(u - u.mean(0, keepdim=True), a - a.mean(0, keepdim=True), "_c", out)
    un = F.normalize(u, dim=-1)
    n = u.size(0)
    out["aniso_u"] = (((un @ un.T).sum() - n) / (n * (n - 1))).item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--adapters", nargs="+", required=True, help="label=path (path 'none' = base model)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = core.read_jsonl(args.val)
    results = {}
    for item in args.adapters:
        label, path = item.split("=", 1)
        model, tok = load(path, dev)
        for k in (3, 0):
            u, a = embed_all(model, tok, rows, k, dev)
            results[f"{label}|k={k}"] = metrics(u, a)
        del model
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    n = len(rows)
    cols = ["cos_pos", "cos_neg", "cos_pos_c", "cos_neg_c", "r1", "r1_c", "r5_c", "aniso_u"]
    print(f"N={n} pairs   chance r1={100 / n:.2f}%  r5={500 / n:.2f}%   embedding offset={OFFSET}")
    print("model|setting".ljust(18) + "  ".join(c.ljust(9) for c in cols))
    for key, m in results.items():
        print(key.ljust(18) + "  ".join(f"{m[c]:.4f}".ljust(9) for c in cols))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
