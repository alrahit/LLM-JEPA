# RESULTS SUMMARY 2026-10-11 (state after round 1 + diagnostics)

## A. Verified results (same script evaluate_synth.py, same frozen test files, all-N denominators)
Parsed accuracy (parse_structural.py: normalized containment), single seed 42:
| | Zero-shot | LM-only run (old "LinAlign") | cosine-JEPA lambda=0.1 run (old "SFT") |
| SYNTH N=2000 | 0.10% (2) | 52.70% (1054) | 50.95% (1019) |
| SPIDER N=1447 | 2.90% (42) | 11.68% (169) | 11.68% (169) |
Raw EM: SYNTH 0 / 1046 / 1010 ; SPIDER 0 / 70 / 70.
Paired LM-only minus cosine-JEPA: SYNTH +1.75 pts, 95% CI [+0.65, +2.90], McNemar p=0.0033 (train-overlap removed: +1.78, p=0.0028). SPIDER 0.00 pts, CI [-0.48, +0.55].
Limits: one seed; parsed metric is string containment, not execution accuracy.

## B. Not re-verified in the final pipeline (old report)
GSM8K 55.57% (733/1319) zero-shot vs 47.54% (627/1319) fine-tuned; Rotten Tomatoes 0.38% (4/1066) vs 91.84% (979/1066). Fine-tuned = LM-only (jepa_loss==0 in all traces). No SFT arm.

## C. What the arms really were
- "LinAlign" runs: InfoNCE at batch size 1 -> 1x1 similarity matrix -> jepa_loss == 0, S == 0, no JEPA gradient (LM-only).
- "SFT" runs: launched without --regular/--infonce/--linalign -> default cosine-JEPA with lambda=0.1 (active).

## D. Diagnostics
1. lambda(t) = exp(-0.025 t) after 20 warmup calls (S==0): below 1e-6 after about 35 optimizer steps, on all four datasets.
2. Controller signals degenerate: linear_map S <= 1/d (~6.5e-4) << tau 0.05; retrieval with B>=2 at the used embedding gives S >= 0.44.
3. Cosine-JEPA collapses final-layer last-token embeddings: cosine ~0.988 for all pairs.
4. Pairing signal lives in mid layers with mean pooling + centering (SYNTH L14 acc16 0.51 vs chance 0.062); final layer last token is at chance.
5. Lexical shortcuts: GSM8K 0.999, SPIDER 0.985 acc16 by word/char overlap; SYNTH partial (0.39 vs model 0.51); Rotten Tomatoes ill-posed (answer = Good/Bad).
6. Engineering: 85-90% of tokens are padding (mean 50/75 tokens vs max_length 512); full-vocab logits for all three views cause OOM at batch>=16; logged loss ~16x (suspected missing accumulation division); bf16 weights + fp16 flag + fp16 eval.
7. Data frozen and hashed; train/test overlap SYNTH 39/2000, SPIDER 3/1447; conclusions unchanged without them.

## E. Earlier ablation (Qwen2.5-1.5B, max_length 128, evaluate.py, one seed; NOT comparable to A)
SYNTH lambda 0/0.5/1.0: 37.85/40.75/43.10%. GSM8K: 35.33/34.72/34.65%. SPIDER: 41/40/41 of 1447.
Pattern: lambda helps only on SYNTH; consistent with the lexical-shortcut explanation.

## F. Open decisions
(1) re-test a correctly wired adaptive lambda (pre-registered), (2) write a diagnostic study, (3) stop. Nothing new has been trained since round 1.
