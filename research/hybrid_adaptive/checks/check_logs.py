import json, glob
base = "hybrid_adaptive/results/qwen/checkpoints"
for name in ["qwen_sft_baseline","qwen_sft_spider","arm1_hybrid_retrieval","spider_hybrid_retrieval"]:
    files = sorted(glob.glob(f"{base}/{name}/**/trainer_state.json", recursive=True), key=lambda f: json.load(open(f))["global_step"])
    st = json.load(open(files[-1]))
    keys = set()
    for e in st["log_history"]: keys |= set(e.keys())
    print(f"== {name}: global_step={st['global_step']} epoch={st['epoch']} best_ckpt={st.get('best_model_checkpoint')}")
    print("   log keys:", sorted(keys))
