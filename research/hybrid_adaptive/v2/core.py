"""core.py - clean building blocks for the LLM-JEPA / LinAlign experiments (v2).

Design rules (each one fixes a problem found in the round-1 audit):
  * labels are built here and are NEVER rebuilt by a collator (only the answer tokens are trained)
  * one code path for every arm: SFT is lam == 0 and the JEPA branch is not executed at all
  * the loss is normalised by the whole accumulation window (exact gradient accumulation)
  * logits (lm_head) are computed only at labelled positions (no full-vocab logits for all tokens)
  * the embedding is taken at a fixed offset from the end (default -3 = last [PRED] token for Qwen)
"""
from __future__ import annotations

import json
from contextlib import nullcontext

import torch
import torch.nn.functional as F

IGNORE = -100
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ----------------------------------------------------------------------------- data
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def add_predictors(text, k):
    """Append <|predictor_k|> ... <|predictor_1|> (same order as the original code)."""
    for j in range(k, 0, -1):
        text += f"<|predictor_{j}|>"
    return text


def _encode(tok, messages, add_generation_prompt):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    return tok(text, add_special_tokens=False)["input_ids"]


def build_example(tok, messages, k_pred, max_length, with_views):
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant"], f"unexpected roles {roles}"
    ids = _encode(tok, messages, add_generation_prompt=False)
    prompt_ids = _encode(tok, messages[:-1], add_generation_prompt=True)
    assert ids[: len(prompt_ids)] == prompt_ids, "prompt is not a prefix of the full chat"
    assert len(ids) <= max_length, f"sequence too long: {len(ids)} > {max_length}"
    labels = [IGNORE] * len(prompt_ids) + ids[len(prompt_ids):]
    ex = {"input_ids": ids, "labels": labels}
    if with_views:
        user = [{"role": "user", "content": add_predictors(messages[1]["content"], k_pred)}]
        asst = [{"role": "assistant", "content": messages[2]["content"]}]
        ex["user_ids"] = _encode(tok, user, add_generation_prompt=False)
        ex["asst_ids"] = _encode(tok, asst, add_generation_prompt=False)
    return ex


def _pad(seqs, pad_value):
    longest = max(len(s) for s in seqs)
    out = torch.full((len(seqs), longest), pad_value, dtype=torch.long)
    mask = torch.zeros((len(seqs), longest), dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1
    return out, mask


def collate(batch, pad_id):
    """Right-padding collator. Labels are padded with IGNORE and are never overwritten."""
    b = {}
    b["input_ids"], b["attention_mask"] = _pad([x["input_ids"] for x in batch], pad_id)
    b["labels"], _ = _pad([x["labels"] for x in batch], IGNORE)
    if "user_ids" in batch[0]:
        b["user_ids"], b["user_mask"] = _pad([x["user_ids"] for x in batch], pad_id)
        b["asst_ids"], b["asst_mask"] = _pad([x["asst_ids"] for x in batch], pad_id)
    return b


# ----------------------------------------------------------------------------- model
def init_predictor_embeddings(model, tok, k):
    """Give the <|predictor_i|> rows a deterministic init (mean of the real vocabulary rows)."""
    names = [f"<|predictor_{i}|>" for i in range(1, k + 1)]
    ids = tok.convert_tokens_to_ids(names)
    if max(ids) >= model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tok))
    emb = model.get_input_embeddings()
    n_real = tok.convert_tokens_to_ids("<|endoftext|>")
    with torch.no_grad():
        mean = emb.weight[:n_real].float().mean(dim=0)
        for i in ids:
            emb.weight[i] = mean.to(emb.weight.dtype)


def load_model_and_tokenizer(model_name, k_pred, lora_rank, dtype, device, grad_ckpt, seed):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if k_pred > 0:
        tok.add_special_tokens({"additional_special_tokens": [f"<|predictor_{i}|>" for i in range(1, k_pred + 1)]})
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.config.use_cache = False
    if k_pred > 0:
        init_predictor_embeddings(model, tok, k_pred)
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    torch.manual_seed(seed)  # LoRA init depends on this seed
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=lora_rank,
                     lora_alpha=2 * lora_rank, lora_dropout=0.1, target_modules=TARGET_MODULES)
    model = get_peft_model(model, cfg)
    model.to(device)
    return model, tok


def inner_lm(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def encode(model, ids, mask):
    """Final-layer hidden states (after the final norm), without computing any logits."""
    return inner_lm(model).model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state


# ----------------------------------------------------------------------------- losses
def lm_loss_sum(model, hidden, labels):
    """Summed next-token cross-entropy, lm_head applied only at labelled positions."""
    h = hidden[:, :-1, :]
    tgt = labels[:, 1:]
    sel = tgt != IGNORE
    n = int(sel.sum().item())
    assert n > 0, "batch has no labelled tokens"
    logits = inner_lm(model).lm_head(h[sel]).float()
    return F.cross_entropy(logits, tgt[sel], reduction="sum"), n


def last_token_embedding(hidden, mask, offset):
    """Hidden state at (length + offset) of every right-padded row. offset=-1 is the last real token."""
    idx = mask.sum(dim=1) + offset
    assert bool((idx >= 0).all()), "offset points before the start of a sequence"
    return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]


def jepa_cosine_sum(u, a):
    cos = F.cosine_similarity(u.float(), a.float(), dim=-1)
    return (1.0 - cos).sum()


@torch.no_grad()
def alignment_diagnostics(u, a):
    un = F.normalize(u.float(), dim=-1)
    an = F.normalize(a.float(), dim=-1)
    sim = un @ an.T
    bsz = sim.size(0)
    out = {"cos_pos": sim.diag().mean().item(), "cos_neg": float("nan"), "retrieval_acc": float("nan")}
    if bsz > 1:
        out["cos_neg"] = ((sim.sum() - sim.diag().sum()) / (bsz * (bsz - 1))).item()
        out["retrieval_acc"] = (sim.argmax(dim=1) == torch.arange(bsz, device=sim.device)).float().mean().item()
    return out


def window_sizes(batches):
    """(number of labelled tokens, number of samples) over a whole accumulation window."""
    tokens = sum(int((b["labels"][:, 1:] != IGNORE).sum().item()) for b in batches)
    samples = sum(b["labels"].size(0) for b in batches)
    return tokens, samples


def micro_step(model, batch, lam, last_token, window_tokens, window_samples, device, autocast_dtype=None):
    """Forward one micro-batch. Returns (loss_to_backward, stats).

    loss = lm_sum / window_tokens + lam * jepa_sum / window_samples, so summing the losses of all
    micro-batches in a window equals the loss of one big batch (exact gradient accumulation).
    lam == 0 means plain SFT: the JEPA branch (extra forward passes) is not executed at all.
    """
    batch = {k: v.to(device) for k, v in batch.items()}
    ctx = torch.autocast(device_type=device.type, dtype=autocast_dtype) if autocast_dtype is not None else nullcontext()
    with ctx:
        hidden = encode(model, batch["input_ids"], batch["attention_mask"])
        lm_sum, n_tok = lm_loss_sum(model, hidden, batch["labels"])
        total = lm_sum / window_tokens
        stats = {"lm_sum": lm_sum.item(), "n_tok": n_tok, "n_samples": int(batch["labels"].size(0))}
        if lam > 0:
            hu = encode(model, batch["user_ids"], batch["user_mask"])
            ha = encode(model, batch["asst_ids"], batch["asst_mask"])
            u = last_token_embedding(hu, batch["user_mask"], last_token)
            a = last_token_embedding(ha, batch["asst_mask"], last_token)
            jsum = jepa_cosine_sum(u, a)
            total = total + lam * jsum / window_samples
            stats["jepa_sum"] = jsum.item()
            stats.update(alignment_diagnostics(u, a))
    return total, stats
