import json, hashlib, glob, os, time, torch

D = "datasets"
names = ["spider_train","spider_test","synth_train","synth_test","synth_test_train","synth_test_test"]
data = {}
print("== 1) Dataset fingerprints")
for n in names:
    p = f"{D}/{n}.jsonl"
    if not os.path.exists(p):
        print(n, "MISSING"); continue
    raw = open(p, "rb").read()
    rows = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    data[n] = rows
    mt = time.strftime('%m/%d %H:%M', time.localtime(os.path.getmtime(p)))
    print(f"{n:18s} rows={len(rows):5d} sha256={hashlib.sha256(raw).hexdigest()[:16]} modified={mt}")

def key(r): return json.dumps(r["messages"][:-1], ensure_ascii=False)

print("\n== 2) Train/test overlap (must be 0)")
for tr, te in [("spider_train","spider_test"),("synth_train","synth_test")]:
    if tr in data and te in data:
        A = {key(r) for r in data[tr]}
        B = [key(r) for r in data[te]]
        print(f"{tr} vs {te}: overlapping test prompts = {sum(k in A for k in B)} / {len(B)} ; unique test prompts = {len(set(B))}")
for n in ["synth_test_train","synth_test_test"]:
    if n in data and "synth_test" in data and "synth_train" in data:
        S = {key(r) for r in data["synth_test"]}
        T = {key(r) for r in data["synth_train"]}
        k = [key(r) for r in data[n]]
        print(f"{n}: inside synth_test = {sum(x in S for x in k)}/{len(k)} ; inside synth_train = {sum(x in T for x in k)}/{len(k)}")

print("\n== 3) Training args per checkpoint")
base = "hybrid_adaptive/results/qwen/checkpoints"
fields = ["num_train_epochs","learning_rate","per_device_train_batch_size","gradient_accumulation_steps","seed","data_seed","lr_scheduler_type","warmup_ratio","weight_decay","bf16","fp16","max_steps","optim"]
for name in ["qwen_sft_baseline","qwen_sft_spider","arm1_hybrid_retrieval","spider_hybrid_retrieval"]:
    cands = glob.glob(f"{base}/{name}/training_args.bin") + sorted(glob.glob(f"{base}/{name}/checkpoint-*/training_args.bin"))
    print(f"-- {name}")
    if not cands:
        print("   no training_args.bin"); continue
    try:
        a = torch.load(cands[-1], weights_only=False)
        print("   " + ", ".join(f"{f}={getattr(a, f, '?')}" for f in fields))
    except Exception as e:
        print("   could not read:", e)
