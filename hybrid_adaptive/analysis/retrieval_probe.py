r"""Offline retrieval-signal probe for LinAlign / LLM-JEPA.

For each model, embeds the user view and the assistant view of test/train examples
exactly like finetune_hybrid.py does (chat template of the single message, last-token
hidden state of the final layer), then measures in-batch retrieval accuracy for several
batch sizes B. This is the signal the LinAlign controller uses (S = 1 - retrieval_acc),
and it needs B >= 2 to be informative.

Run from the repo root (D:/LLM_JEPA):
  python hybrid_adaptive\analysis\retrieval_probe.py --dataset synth
"""
import argparse, gc, json, os, time
import torch
import torch.nn.functional as F

BATCH_SIZES = [2, 4, 8, 16, 32, 64]


def read_rows(path, n):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


def views(messages):
    """Same views as finetune_hybrid.py for non-gemma models: user = messages[1:2], assistant = messages[2:3]."""
    return [messages[1]], [messages[2]]


def retrieval_stats(U, A, batch_sizes=BATCH_SIZES, tau=0.07, seed=0):
    """U, A: [N, d] tensors, row i of U is paired with row i of A."""
    U = F.normalize(U.float(), dim=1)
    A = F.normalize(A.float(), dim=1)
    N = U.shape[0]
    full = U @ A.T
    eye = torch.eye(N, dtype=torch.bool)
    res = {
        "N": N,
        "pos_cos": full[eye].mean().item(),
        "neg_cos": full[~eye].mean().item(),
        "by_batch": {},
    }
    g = torch.Generator().manual_seed(seed)
    for B in batch_sizes:
        if B > N:
            continue
        accs, accs_rev, losses = [], [], []
        for rep in range(3):  # three random partitions for stability
            perm = torch.randperm(N, generator=g)
            for s in range(0, N - B + 1, B):
                idx = perm[s:s + B]
                sim = U[idx] @ A[idx].T
                tgt = torch.arange(B)
                accs.append((sim.argmax(1) == tgt).float().mean().item())
                accs_rev.append((sim.argmax(0) == tgt).float().mean().item())
                losses.append(F.cross_entropy(sim / tau, tgt).item())
        m = lambda x: sum(x) / len(x)
        res["by_batch"][B] = {
            "acc_user_to_assistant": m(accs),
            "acc_assistant_to_user": m(accs_rev),
            "chance": 1.0 / B,
            "infonce_loss": m(losses),
            "S_mean": 1.0 - m(accs),
        }
    return res


@torch.inference_mode()
def embed_all(model, tok, rows, device):
    us, as_ = [], []
    for r in rows:
        um, am = views(r["messages"])
        for msgs, store in ((um, us), (am, as_)):
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            enc = tok(text, return_tensors="pt").to(device)
            out = model(**enc, output_hidden_states=True)
            store.append(out.hidden_states[-1][0, -1].float().cpu())
    return torch.stack(us), torch.stack(as_)


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
    splits = {s: read_rows(f"datasets/{args.dataset}_{s}.jsonl", args.n) for s in ("train", "test")}
    device = "cuda"
    report = {}
    for label, path in models:
        print(f"\n##### {label}  [{path}]")
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=device)
        model.eval()
        report[label] = {}
        for split, rows in splits.items():
            t = time.time()
            U, A = embed_all(model, tok, rows, device)
            st = retrieval_stats(U, A)
            report[label][split] = st
            print(f"  [{split}] N={st['N']}  pos_cos={st['pos_cos']:.3f}  neg_cos={st['neg_cos']:.3f}  ({time.time()-t:.0f}s)")
            print("     B   acc(u->a)  acc(a->u)  chance   InfoNCE(tau=.07)   S=1-acc")
            for B, v in st["by_batch"].items():
                print(f"    {B:3d}   {v['acc_user_to_assistant']:.3f}      {v['acc_assistant_to_user']:.3f}     {v['chance']:.3f}      {v['infonce_loss']:.3f}          {v['S_mean']:.3f}")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    os.makedirs("hybrid_adaptive/results/diagnostics", exist_ok=True)
    out = f"hybrid_adaptive/results/diagnostics/retrieval_probe_{args.dataset}.json"
    json.dump(report, open(out, "w"), indent=2)
    print("\nsaved:", out)
    print("Controller premise: lambda only decays if S_mean < tau (0.05), i.e. retrieval accuracy > 95%.")


if __name__ == "__main__":
    main()
