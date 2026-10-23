"""CPU unit tests for core.py (tiny random Qwen2 + the real tokenizer). Run from hybrid_adaptive/."""
import os, sys, traceback

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoTokenizer, Qwen2Config, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
import core

MODEL = "Qwen/Qwen2-1.5B-Instruct"
K = 3
tok = AutoTokenizer.from_pretrained(MODEL)
tok.add_special_tokens({"additional_special_tokens": [f"<|predictor_{i}|>" for i in range(1, K + 1)]})
PAD = tok.pad_token_id

ROWS = None
for base in (".", ".."):
    p = os.path.join(base, "datasets", "spider_train.jsonl")
    if os.path.exists(p):
        ROWS = core.read_jsonl(p)[:6]
        break
if ROWS is None:
    ROWS = [{"messages": [
        {"role": "system", "content": "Convert natural language to SQL."},
        {"role": "user", "content": f"For db_id:[d{i}]\n\nHow many items are there in table {i}?"},
        {"role": "assistant", "content": f"SELECT count(*) FROM t{i};"}]} for i in range(6)]


def make_model():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      tie_word_embeddings=True)
    m = AutoModelForCausalLM.from_config(cfg)
    m = get_peft_model(m, LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8, lora_dropout=0.0,
                                     target_modules=core.TARGET_MODULES))
    for n, p in m.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.05)
    m.eval()
    return m


def batch_of(rows, k=K, views=True):
    exs = [core.build_example(tok, r["messages"], k, 512, views) for r in rows]
    return core.collate(exs, PAD)


def test_labels_mask_prompt_and_train_answer():
    for r in ROWS:
        ex = core.build_example(tok, r["messages"], K, 512, True)
        lab = [t for t in ex["labels"] if t != core.IGNORE]
        n_prompt = len(ex["labels"]) - len(lab)
        assert all(t == core.IGNORE for t in ex["labels"][:n_prompt])
        assert ex["labels"][n_prompt:] == ex["input_ids"][n_prompt:]
        text = tok.decode(lab)
        assert text.startswith(r["messages"][2]["content"]), text
        assert "<|im_end|>" in text, "the model must learn to stop"


def test_collate_keeps_labels_and_masks():
    b = batch_of(ROWS[:3])
    assert b["labels"].shape == b["input_ids"].shape
    short = int(b["attention_mask"].sum(dim=1).argmin())
    n = int(b["attention_mask"][short].sum())
    assert bool((b["labels"][short, n:] == core.IGNORE).all())
    assert bool((b["labels"][:, 0] == core.IGNORE).all())


def test_embedding_token_is_last_predictor_and_last_content_token():
    for r in ROWS[:3]:
        ex = core.build_example(tok, r["messages"], K, 512, True)
        assert tok.decode([ex["user_ids"][-3]]) == "<|predictor_1|>"
        assert tok.decode([ex["user_ids"][-2]]) == "<|im_end|>"
        assert tok.decode([ex["asst_ids"][-2]]) == "<|im_end|>"
        assert r["messages"][2]["content"].endswith(tok.decode([ex["asst_ids"][-3]]))


def test_lm_loss_matches_hf_loss():
    m = make_model()
    b = batch_of(ROWS[:4], views=False)
    with torch.no_grad():
        ref = m(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"]).loss
        h = core.encode(m, b["input_ids"], b["attention_mask"])
        s, n = core.lm_loss_sum(m, h, b["labels"])
    assert abs(float(ref) - float(s) / n) < 1e-5, (float(ref), float(s) / n)


def test_embedding_independent_of_padding():
    m = make_model()
    b = batch_of(ROWS[:4])
    with torch.no_grad():
        hu = core.encode(m, b["user_ids"], b["user_mask"])
        e_batch = core.last_token_embedding(hu, b["user_mask"], -3)
        single = batch_of(ROWS[:1])
        hs = core.encode(m, single["user_ids"], single["user_mask"])
        e_single = core.last_token_embedding(hs, single["user_mask"], -3)
    assert torch.allclose(e_batch[0], e_single[0], atol=1e-5)


def test_lambda_zero_skips_jepa_and_has_same_lm_part():
    m = make_model()
    b = batch_of(ROWS[:4])
    dev = torch.device("cpu")
    nt, ns = core.window_sizes([b])
    only_lm = {k: v for k, v in b.items() if k in ("input_ids", "attention_mask", "labels")}
    _, s0 = core.micro_step(m, only_lm, 0.0, -3, nt, ns, dev)   # must not need the view tensors
    assert "jepa_sum" not in s0
    _, s1 = core.micro_step(m, b, 1.0, -3, nt, ns, dev)
    assert "jepa_sum" in s1
    assert abs(s0["lm_sum"] - s1["lm_sum"]) < 1e-4


def _grads(m, batches, lam):
    m.zero_grad()
    nt, ns = core.window_sizes(batches)
    for b in batches:
        loss, _ = core.micro_step(m, b, lam, -3, nt, ns, torch.device("cpu"))
        loss.backward()
    return {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}


def test_gradient_accumulation_is_exact():
    m = make_model()
    full = _grads(m, [batch_of(ROWS[:4])], lam=1.0)
    split = _grads(m, [batch_of(ROWS[:2]), batch_of(ROWS[2:4])], lam=1.0)
    assert full.keys() == split.keys() and len(full) > 0
    worst = max(float((full[k] - split[k]).abs().max()) for k in full)
    assert worst < 1e-5, worst


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS", name)
        except Exception:
            failed += 1
            print("FAIL", name)
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
