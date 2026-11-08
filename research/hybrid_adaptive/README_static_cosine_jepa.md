# Static cosine-JEPA fine-tuning: a pre-registered replication (negative result so far)

Branch: `research/static-cosine-jepa`. Pre-registration: `PREREGISTRATION_v1.md` (tag `prereg-v1`, amendments 1 to 12). Code: `v2/`.

## Question
Does adding a static cosine-JEPA loss to supervised fine-tuning of a language model (a) improve task accuracy, and (b) make the user-view and assistant-view embeddings more linearly related?

## Method
Loss = LM loss + lambda x (1 - cosine) between the user-view and assistant-view last-token embeddings (`--last_token -3`), with k predictor tokens. Run with `v2/train.py --arm static --lam LAMBDA --k K`. The control arm is LM-only (`--arm sft`).
## Setup
Qwen/Qwen2-1.5B-Instruct (revision ba1cf1846d7df0a0591d6c00649f57e798519da8), LoRA rank 32, bf16, seeds 82, 23, 37. Datasets measured: Synth and Turk (test files hash-checked against the run args). Selection of lambda, k and learning rate on a pilot split only (Stage 1), then full training and test (Stage 2). Paired bootstrap, 10,000 resamples. Success rule for H0: mean paired difference of at least 1.0 point with the 95% interval above 0. H1: JEPA has higher held-out linearity than LM-only, interval above 0 on at least two datasets.
## Results
| Test | Synth | Turk |
|---|---|---|
| Accuracy, JEPA minus LM-only (points, 95% interval) | +0.317 [-0.100, +0.733] | +0.100 [-0.667, +0.850] |
| H0 (at least 1.0 pt, interval above 0) | not met | not met |
| Held-out linearity, JEPA minus LM-only (95% interval) | -0.1858 [-0.1976, -0.1738] | -0.1731 [-0.1852, -0.1610] |
| H1 (positive, interval above 0) | not met | not met |
| Mean cosine between the two views, JEPA arm | 0.9999 | 1.0000 |

Collapse: all 22 static runs reached cos_pos and cos_neg of about 1. Lower learning rates (4e-5, 1e-5) and k = 3 delayed the collapse within 200 steps but did not prevent it (amendments 10 and 11).
## Why this does not reach the goal
In this setup the cosine loss is satisfied by collapsing both views to almost the same point (user-view variance about 2.2 for JEPA against about 1e4 for LM-only), accuracy does not improve by the pre-registered margin, and the held-out linear relation between the views is weaker than with LM-only training. We did not test whether preventing collapse would give a gain; that is not claimed either way.

## Limits
One model size, LoRA only, two of the eight planned datasets, three seeds, embeddings in bf16 (noise floor of the collapsed arm not measured), no loss dropout, no shuffled-pair control, no predictor-token or linearity-only ablation (H2 not tested). Spider, the other datasets and Llama models are not run. These results do not show that JEPA never helps; they show that this static cosine recipe did not help here.

## Files
- `PREREGISTRATION_v1.md`: protocol, selection rules and all outcomes, amendments 1 to 12.
- `v2/`: training, scoring, `dump_embeddings.py`, `linearity_v2.py`, analysis scripts.