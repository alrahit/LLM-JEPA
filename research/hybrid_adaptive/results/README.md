# results/ guide

All evaluations use scripts/evaluate_synth.py (greedy, 256 new tokens, fp16) on the frozen test files.
"LM-only run" = checkpoint trained with --infonce --linalign at batch size 1 (JEPA gradient exactly zero).
"cosine-JEPA run" = checkpoint trained without --regular/--infonce/--linalign (default cosine-JEPA, lambda 0.1).

## Final, same-pipeline results (use these)
| File | Model |
|---|---|
| final_synth_zeroshot_eval_results.json, final_spider_zeroshot_eval_results.json | Qwen2-1.5B-Instruct, zero-shot |
| final_synth_linalign_eval_results.json, final_spider_linalign_eval_results.json | LM-only run (checkpoints arm1_hybrid_retrieval, spider_hybrid_retrieval) |
| final_synth_sft_eval_results.json, spider_sft_true_eval_results.json | cosine-JEPA run (checkpoints qwen_sft_baseline, qwen_sft_spider) |

## Earlier files (reproduced exactly by the final ones; kept for provenance)
- qwen/synth_eval_results.json (2026-09-19) and qwen/spider_eval_results.json (2026-09-20): first evaluation of the LM-only runs.
- qwen/spider_baseline_eval_results.json, qwen/evaluations/*: zero-shot and older evaluations, including GSM8K and Rotten Tomatoes (not re-run in the final pipeline).

## Known mislabeled / misleading files (do not use for conclusions)
- qwen/evaluations/spider/spider_sft_eval_results.json and spider_eval_results.json: byte-identical copies of the LM-only SPIDER run (same sha256), NOT an SFT result.
- qwen/evaluations/spider/spider_final_summary.txt: labels the LM-only raw EM (4.84%) as "SFT" and LinAlign as pending.
- qwen/evaluations/synth/synth_final_summary.txt: compares a parsed LinAlign number (52.70) with a raw-EM SFT number (50.50); the like-for-like parsed values are 52.70 vs 50.95.
- qwen/evaluations/synth/synth_eval_results.json holds the cosine-JEPA SYNTH run (50.5%), although the file name suggests LinAlign.

## Traces and diagnostics
- qwen/traces/*_trace.json: controller histories (16 micro-steps are logged at each logging step). jepa_loss and S are 0 in every entry.
- diagnostics/retrieval_probe_synth.json, retrieval_probe2_synth.json, retrieval_probe2_spider.json: in-batch retrieval accuracy by batch size, layer, token position, pooling and centering.

## Not in git
Model checkpoints (results/qwen/checkpoints/) are ignored because of size.
