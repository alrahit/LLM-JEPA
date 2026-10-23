import json
import numpy as np
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('hybrid_adaptive/results/qwen/checkpoints/qwen_sft_baseline')
for name in ['synth_train', 'synth_test', 'spider_train', 'spider_test']:
    rows = [json.loads(l) for l in open(f'datasets/{name}.jsonl', encoding='utf-8') if l.strip()]
    full, user, ans = [], [], []
    for r in rows:
        m = r['messages']
        full.append(len(tok(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=False))['input_ids']))
        user.append(len(tok(m[-2]['content'])['input_ids']))
        ans.append(len(tok(m[-1]['content'])['input_ids']))
    f = np.array(full)
    print(f"{name:13s} n={len(f)} full tokens: mean={f.mean():.0f} p50={np.percentile(f,50):.0f} p95={np.percentile(f,95):.0f} p99={np.percentile(f,99):.0f} max={f.max()} | truncated(>512)={(f>512).sum()} | prompt mean={np.mean(user):.0f} answer mean={np.mean(ans):.0f}")
