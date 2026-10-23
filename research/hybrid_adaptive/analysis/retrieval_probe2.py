r"""Probe 2: where in the network is the user<->assistant pairing signal?

For every model, embeds the user view and the assistant view of test examples and
measures in-batch retrieval accuracy (B=16 and B=64) for combinations of:
  layer   : 4 depths + the final layer
  token   : last token (-1), second-last token (-2, <|im_end|> for Qwen), mean over content tokens
  centering: raw vs dataset-mean-centered

Run from the repo root (D:/LLM_JEPA):
  python hybrid_adaptive\analysis\retrieval_probe2.py --dataset synth
"""
import argparse, gc, json, os, time
import torch
import torch.nn.functional as F


def read_rows(path, n):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


@torch.inference_mode()
def embed_all(model, tok, rows, device):
    store = {"u": {}, "a": {}}
    for r in rows:
        for key, msg in (("u", r["messages"][1]), ("a", r["messages"][2])):
            text = tok.apply_chat_template([msg], tokenize=False, add_generation_prompt=False)
            s = text.rfind(msg["content"])
            e = s + len(msg["content"])
            enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
            offs = enc.pop("offset_mapping")[0]
            enc = enc.to(device)
            hs = model(**enc, output_hidden_states=True).hidden_states
            n = len(hs) - 1
            layers = sorted({n // 4, n // 2, (3 * n) // 4, n - 2, n})
            keep = (offs[:, 0] >= s) & (offs[:, 1] <= e) & (offs[:, 1] > offs[:, 0])
            if int(keep.sum()) == 0:  # fallback: last two tokens
                keep = torch.zeros_like(keep)
                keep[-2:] = True
            keep = keep.to(device)
            for L in layers:
                h = hs[L][0].float()
                for pool, vec in (("last", h[-1]), ("imend", h[-2]), ("mean", h[keep].mean(0))):
                    store[key].setdefault((L, pool), []).append(vec.cpu())
    return store


def prep(U, A, centered):
    U = U.float()
    A = A.float()
    if centered:
        U = U - U.mean(0, keepdim=True)
        A = A - A.mean(0, keepdim=True)
    return F.normalize(U, dim=1), F.normalize(A, dim=1)


def acc_at(U, A, B, reps=3, seed=0):
    N = U.shape[0]
    g = torch.Generator().manual_seed(seed)
    hits = []
    for _ in range(reps):
        perm = torch.randperm(N, generator=g)
        for s in range(0, N - B + 1, B):
            idx = perm[s:s + B]
            sim = U[idx] @ A[idx].T
            hits.append((sim.argmax(1) == torch.arange(B)).float().mean().item())
    return sum(hits) / len(hits)


def gap(U, A):
    N = U.shape[0]
    full = U @ A.T
    pos = full.diagonal().mean().item()
    neg = (full.sum() - full.diagonal().sum()).item() / (N * N - N)
    return pos, neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["synth", "spider"])
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--base", default="Qwen/Qwen2-1.5B-Instruct")
    ap.add_argument("--ckpt_root", default="hybrid_adaptive/results/qwen/checkpoints")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    names = {
        "synth": ("arm1_hybrid_retrieval", "qwen_sft_baseline"),
        "spider": ("spider_hybrid_retrieval", "qwen_sft_spider"),
    }[args.dataset]
    models = [
        ("base (zero-shot)", args.base),
        ("LM-only run (old 'LinAlign')", os.path.join(args.ckpt_root, names[0])),
        ("cosine-JEPA run (old 'SFT')", os.path.join(args.ckpt_root, names[1])),
    ]
    rows = read_rows(f"datasets/{args.dataset}_test.jsonl", args.n)
    report = {}
    for label, path in models:
        print(f"\n##### {label}  [{path}]")
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
        model.eval()
        t = time.time()
        store = embed_all(model, tok, rows, "cuda")
        print(f"  embedded {len(rows)} pairs in {time.time()-t:.0f}s")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        n_final = max(k[0] for k in store["u"])
        results = []
        for key in store["u"]:
            Uall = torch.stack(store["u"][key])
            Aall = torch.stack(store["a"][key])
            for centered in (False, True):
                U, A = prep(Uall, Aall, centered)
                pos, neg = gap(U, A)
                results.append({
                    "layer": key[0], "pool": key[1], "centered": centered,
                    "acc16": acc_at(U, A, 16), "acc64": acc_at(U, A, 64),
                    "pos_cos": pos, "neg_cos": neg,
                })
        report[label] = results
        base_row = [r for r in results if r["layer"] == n_final and r["pool"] == "last" and not r["centered"]][0]
        print("  chance: acc16=0.062  acc64=0.016")
        print("  layer  pool    centered  acc16   acc64   pos_cos  neg_cos")
        def show(r, tag=""):
            print(f"  {r['layer']:5d}  {r['pool']:6s}  {str(r['centered']):8s}  {r['acc16']:.3f}   {r['acc64']:.3f}   {r['pos_cos']:.3f}    {r['neg_cos']:.3f}   {tag}")
        show(base_row, "<- what training used (final layer, last token, raw)")
        for r in sorted(results, key=lambda x: -x["acc64"])[:8]:
            show(r)

    os.makedirs("hybrid_adaptive/results/diagnostics", exist_ok=True)
    out = f"hybrid_adaptive/results/diagnostics/retrieval_probe2_{args.dataset}.json"
    json.dump(report, open(out, "w"), indent=2)
    print("\nsaved:", out)


if __name__ == "__main__":
    main()
