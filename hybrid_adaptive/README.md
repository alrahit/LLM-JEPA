# hybrid_adaptive: LinAlign experiments (round 1 + diagnostics)

Status (2026-10-11): round 1 is complete and re-verified. The implemented LinAlign controller was inert
(see "Key findings"); the claims of the original report are not supported by this round. Nothing new has been
trained since. See docs/RESULTS_SUMMARY_2026-10-11.md for the full state and docs/PREREG_DRAFT.md for the
pre-registration draft of a corrected experiment.

## Verified results (single seed 42; parsed accuracy over all test samples)
| Dataset | Zero-shot | LM-only run | cosine-JEPA (lambda 0.1) run |
|---|---|---|---|
| SYNTH (N=2000) | 0.10% | 52.70% | 50.95% |
| SPIDER (N=1447) | 2.90% | 11.68% | 11.68% |

Paired LM-only minus cosine-JEPA: SYNTH +1.75 pts (95% CI +0.65..+2.90, McNemar p=0.0033); SPIDER 0.00 pts (CI -0.48..+0.55).
Metric is normalized string containment (parse_structural.py), not SQL execution accuracy.
Run names in older files: "LinAlign" = LM-only run, "SFT" = cosine-JEPA lambda 0.1 run (explained below).

## Key findings
1. InfoNCE was computed at per-device batch size 1: the 1x1 similarity matrix gives jepa_loss == 0 and S == 0, so the
   "LinAlign" runs received no JEPA gradient and lambda decayed as exp(-0.025 t) (below 1e-6 after ~35 optimizer steps).
2. The "SFT" runs were launched without --regular/--infonce/--linalign and therefore used the default cosine-JEPA (lambda 0.1).
3. Both controller signals are degenerate as configured (linear_map: S <= 1/d; retrieval at B>=2 on final-layer last token: S >= 0.44).
4. Cosine-JEPA collapses final-layer last-token embeddings (cosine ~0.988 for all pairs).
5. User/assistant pairing signal lives in mid layers (mean pooling + centering), not in the final-layer last token.
6. Pairing is solvable by lexical overlap on SPIDER (0.985) and GSM8K (0.999); Rotten Tomatoes is ill-posed for in-batch InfoNCE.

## Layout (run all commands from the repository root)
| Path | Content |
|---|---|
| finetune_hybrid.py, linalign.py, evaluate_hybrid.py | training / controller / evaluation code used for round 1 (unchanged; sha256 in docs/WORKLOG_2026-10-10.md) |
| scripts/ | helper scripts (evaluate_synth.py, parsers) |
| analysis/ | final_table.py (paired tests, CIs), compare_paired.py, agree.py, retrieval_probe.py, retrieval_probe2.py, lexical_baseline.py |
| checks/ | audit_data.py (hashes, train/test overlap, training args), check_logs.py, token_stats.py, trace_scan.py, trace_vs_loss.py, trace_check2.py |
| tools/ | run_final_evals.ps1 (5 evals, same script and settings), bench.py (memory benchmark) |
| results/ | evaluation outputs, traces, diagnostics (see results/README.md) |
| docs/ | WORKLOG, RESULTS_SUMMARY, PREREG_DRAFT, dataset hashes |

## Reproduce
1. Datasets are frozen; verify with docs/dataset_hashes_2026-10-09.txt (sha256).
2. Evaluation: tools/run_final_evals.ps1 (greedy decoding, 256 new tokens, fp16). Needs the trained checkpoints (not in git).
3. Analysis: python hybrid_adaptive/analysis/final_table.py
4. Diagnostics: python hybrid_adaptive/analysis/retrieval_probe2.py --dataset synth|spider ; python hybrid_adaptive/analysis/lexical_baseline.py --dataset synth|spider|gsm8k|rotten_tomatoes

## Known issues / limitations
- One seed; no seed variance. Parsed metric is not execution accuracy.
- Train/test prompt overlap: SYNTH 39/2000, SPIDER 3/1447 (conclusions unchanged without them).
- Mixed precision: bf16 weights, fp16 training flag, fp16 evaluation. Logged loss is ~16x the micro-batch loss (suspected missing gradient-accumulation division; unverified).
- Padding to max_length 512 although mean length is 50 (SYNTH) / 75 (SPIDER) tokens.

## Attribution
Built on the LLM-JEPA code base; keep the upstream license and notices (see the repository root).
