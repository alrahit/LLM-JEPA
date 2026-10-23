"""train.py - one training loop for every arm.

  --arm sft     : lam is exactly 0, the JEPA branch is never executed
  --arm static  : fixed lam > 0 with k predictor tokens (the original LLM-JEPA recipe)

Never deletes anything: it refuses to start if --out_dir is not empty.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from torch.utils.data import DataLoader

import core


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["sft", "static"], required=True)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--k", type=int, default=0, help="number of [PRED] tokens (static arm only)")
    ap.add_argument("--last_token", type=int, default=-3)
    ap.add_argument("--model", default="Qwen/Qwen2-1.5B-Instruct")
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=82)
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--micro_bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=0, help="stop early after N optimizer steps (smoke test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--no_grad_ckpt", action="store_true")
    ap.add_argument("--save_every_epoch", action="store_true")
    a = ap.parse_args()
    if a.arm == "sft":
        if a.lam not in (None, 0.0):
            sys.exit("ERROR: --arm sft means lam == 0 exactly (got --lam %s)" % a.lam)
        a.lam, a.k = 0.0, 0
    else:
        if a.lam is None or a.lam <= 0:
            sys.exit("ERROR: --arm static needs --lam > 0")
        if a.micro_bs < 2:
            print("WARNING: micro_bs < 2, in-batch alignment diagnostics will be NaN")
    if a.device == "cpu":
        a.dtype = "fp32"
    return a


def main():
    args = parse()
    if os.path.exists(args.out_dir) and os.listdir(args.out_dir):
        sys.exit(f"ERROR: out_dir is not empty: {args.out_dir} (refusing to overwrite, pick a new --out_dir)")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    use_bf16 = args.dtype == "bf16"
    if use_bf16 and dev.type == "cuda" and not torch.cuda.is_bf16_supported():
        sys.exit("ERROR: this GPU does not support bf16")
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    autocast_dtype = torch.bfloat16 if use_bf16 else None

    model, tok = core.load_model_and_tokenizer(args.model, args.k, args.lora_rank, dtype, dev,
                                               grad_ckpt=not args.no_grad_ckpt, seed=args.seed)
    rows = core.read_jsonl(args.train_file)
    with_views = args.arm == "static"
    examples = [core.build_example(tok, r["messages"], args.k, args.max_length, with_views) for r in rows]
    g = torch.Generator()
    g.manual_seed(args.seed)  # same data order for every arm with the same seed
    pad_id = tok.pad_token_id
    loader = DataLoader(examples, batch_size=args.micro_bs, shuffle=True, drop_last=True, generator=g,
                        collate_fn=lambda b: core.collate(b, pad_id))
    steps_per_epoch = len(loader) // args.accum
    total_steps = steps_per_epoch * args.epochs
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1.0 - s / total_steps))
    print(f"arm={args.arm} lam={args.lam} k={args.k} lr={args.lr} rank={args.lora_rank} seed={args.seed}")
    print(f"examples={len(examples)} micro_bs={args.micro_bs} accum={args.accum} "
          f"steps/epoch={steps_per_epoch} total_steps={total_steps} trainable_params={n_train / 1e6:.1f}M", flush=True)

    model.train()
    log_f = open(os.path.join(args.out_dir, "train_log.jsonl"), "a", encoding="utf-8")
    step, t0, stop = 0, time.time(), False
    last = {}
    for epoch in range(1, args.epochs + 1):
        it = iter(loader)
        for _ in range(steps_per_epoch):
            window = [next(it) for _ in range(args.accum)]
            nt, ns = core.window_sizes(window)
            sums, diag = defaultdict(float), defaultdict(list)
            for b in window:
                loss, st = core.micro_step(model, b, args.lam, args.last_token, nt, ns, dev, autocast_dtype)
                if not torch.isfinite(loss):
                    sys.exit(f"ERROR: non-finite loss at step {step + 1}")
                loss.backward()
                sums["lm"] += st["lm_sum"]
                sums["jepa"] += st.get("jepa_sum", 0.0)
                for key in ("cos_pos", "cos_neg", "retrieval_acc"):
                    if key in st:
                        diag[key].append(st[key])
            gn = float(torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm))
            lr_now = opt.param_groups[0]["lr"]
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            last = {"step": step, "epoch": epoch, "lr": lr_now, "lam": args.lam, "lm_loss": sums["lm"] / nt,
                    "grad_norm": gn, "elapsed_s": round(time.time() - t0, 1)}
            if args.lam > 0:
                last["jepa_loss"] = sums["jepa"] / ns
                for key, vals in diag.items():
                    last[key] = sum(vals) / len(vals)
            if dev.type == "cuda":
                last["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            if step == 1 or step % args.log_every == 0 or step == total_steps:
                log_f.write(json.dumps(last) + "\n")
                log_f.flush()
                shown = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in last.items()}
                print(shown, flush=True)
            if args.max_steps and step >= args.max_steps:
                stop = True
                break
        if args.save_every_epoch or epoch == args.epochs or stop:
            model.save_pretrained(os.path.join(args.out_dir, f"adapter_epoch{epoch}"))
        if stop:
            break
    secs = time.time() - t0
    samples = step * args.micro_bs * args.accum
    with open(os.path.join(args.out_dir, "final.json"), "w") as f:
        json.dump({"steps": step, "seconds": round(secs, 1), "samples_per_s": round(samples / max(secs, 1e-9), 2),
                   "last": last}, f, indent=1)
    print(f"done: {step} steps, {secs:.0f}s, {samples / max(secs, 1e-9):.2f} samples/s", flush=True)


if __name__ == "__main__":
    main()
