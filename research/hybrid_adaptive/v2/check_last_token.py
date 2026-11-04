import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2-1.5B-Instruct")
ap.add_argument("--file", required=True)
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--offsets", type=int, nargs="+", default=[-1, -2, -3, -4])
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
if a.k > 0:
    tok.add_special_tokens({"additional_special_tokens": [f"<|predictor_{i}|>" for i in range(1, a.k + 1)]})
for i, r in enumerate(core.read_jsonl(a.file)[:a.n]):
    ex = core.build_example(tok, r["messages"], a.k, 4096, True)
    print(f"\n--- sample {i}")
    for name, ids in (("user view", ex["user_ids"]), ("assistant view", ex["asst_ids"])):
        print(f"{name} (length {len(ids)})")
        for off in a.offsets:
            print(f"   offset {off:>2}: id {ids[off]:>7}  token {tok.decode([ids[off]])!r}")
