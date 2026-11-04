# Pre-registration v1: LLM-JEPA replication and mechanism study

Project: `alrahit/LLM-JEPA`, branch `research/experimental-results-2026`. Date: 4 November 2026.
Code frozen at tag `v2-prereg-2` (commit `66177dc`). This file is committed and tagged `prereg-v1`.

## 0. Status and honesty statement
- Written **after** these exploratory results: (a) round 1 (Qwen2-1.5B-Instruct, LoRA, learning rate about 1e-5; scored on the Synth, Spider and other test files); (b) an earlier Qwen2.5 sweep on Synth (max_length 128, one seed); (c) 13 v2 pilot runs on Spider validation with Qwen2 (LM-only and static JEPA, k = 3). These informed the learning-rate grid and the decision to treat round 1 as undertrained. They are reported as exploratory.
- At the time of writing there is **no** Stage 1 or Stage 2 run of the protocol below, and no Llama run.
- Round-1 models were scored on test files. No setting below was chosen from those test scores.

## 1. Hypotheses and decision rules
| ID | Hypothesis | Rule |
|---|---|---|
| H0 | Static cosine-JEPA beats LM-only on Synth and Turk (primary), Spider and the other datasets (secondary) | Mean paired difference of at least 1.0 point with the 95% interval above 0 |
| H1 | JEPA models have higher held-out linearity than LM-only models | Positive difference, 95% interval above 0, on at least two datasets |
| H2 | Linearity is more than a by-product of alignment | Shuffled-pair JEPA gives no accuracy or linearity gain, and a k > 0 predictor or a linearity-only auxiliary loss changes accuracy in the predicted direction; otherwise report "association only" |
| H3 | Gains depend on the task | Report the correlation between gain per dataset and a word-overlap pairing baseline, with its interval; no causal wording |
| H4 (conditional) | A self-tuning lambda controller matches the best static lambda | Within 1.0 point on every dataset with fixed defaults. Run only if the controller signal passes the offline checks in section 9 |

The 1.0-point margins come from the project's earlier draft.

## 2. Models
- Primary: `meta-llama/Llama-3.2-1B-Instruct`. Second family: `Qwen/Qwen2-1.5B-Instruct`.
- Revision hashes, the Llama chat-template date handling and the Llama last-token offset are recorded in the amendment log (section 11) before the first run of each model.
- Qwen2 last-token offset -3 is verified (k = 3). Llama is expected to need -2; this is an expectation until checked.

## 3. Data
Eight datasets: Synth, Turk, Spider, GSM8K, Rotten Tomatoes, Yelp, NQ-Open, HellaSwag (`paraphrase_train.jsonl` is used only for pretraining experiments).
- **Stage 1 (select settings):** train on `pilot_train.jsonl`, score on `pilot_val.jsonl` (made by `make_splits.py`, seed 1234, 5% validation, one output folder per dataset).
- **Stage 2 (final):** train on the full `*_train.jsonl` with the chosen settings, score on `*_test.jsonl`. Validation rows are inside the Stage 2 training data; validation is never reported.
- Both stages are the same for every arm.

Split hashes (first 16 hex of SHA-256):
| Dataset | Total | Train | Val | sha_train | sha_val |
|---|---|---|---|---|---|
| gsm8k | 7473 | 7099 | 374 | a48011d2ac8ad498 | 065fcf1c276d7d17 |
| hellaswag | 8042 | 7640 | 402 | a2ee8b58289efb12 | 24f78013124270ce |
| nq_open | 8000 | 7600 | 400 | 2dbcedd47f2b6fe4 | 4a25d5dddeb2c437 |
| rotten_tomatoes | 8530 | 8104 | 426 | 2e61fef321c6ad7d | 392c27f5cbd15d79 |
| spider | 6587 | 6258 | 329 | b8c9f77acb7b510c | 9411276309faec2a |
| synth | 8000 | 7600 | 400 | 48027baf62e8c394 | 315694d2f5035d07 |
| turk | 8000 | 7600 | 400 | ceccf47caa7cb903 | 171d3774dd19d525 |
| yelp | 8000 | 7600 | 400 | d36c829165ad11a2 | ef363b9443c51823 |

Test files: gsm8k 9D0C4C943E0D770B, hellaswag 86EE4865CAAE4E5A, nq_open 5689F09A771A31AD, rotten_tomatoes B5E55EDB73ECFE26, spider 5ABFB4E88FB549E5, synth F94460595708F166, turk CC6C74BCB005386C, yelp F3A9908DDDFB696E.
Full training files: gsm8k A9FE4884AFE879E5, hellaswag EE95DF64650519C8, nq_open D3751B48AB3927AA, paraphrase 0DC5A1DC6B029BF4, rotten_tomatoes 19F91E1EE3CE9711, spider 9D22697FE4CAA902, synth DA2D42CC506BF4D9, turk 21177E6702E53E41, yelp 3F40ACCEAFF085EA.

Each run records the SHA-256 of its train, validation and test files in `args.json` (`data_sha256`). A run whose hashes differ from the above is excluded.

## 4. Training and selection
- LoRA rank 32, bf16, 4 epochs, micro-batch 4 with accumulation 4 (effective batch 16), `max_length` 512, linear decay, gradient checkpointing on (the `train.py` defaults at tag `v2-prereg-2`).
- Stage 1, learning rate: LM-only arm, seed 82, grid {2e-4, 4e-4, 8e-4}, chosen by the validation primary metric. A tie within 0.5 points goes to the smaller rate. Repeated for each model and dataset.
- Stage 1, JEPA settings: the chosen learning rate is reused. k in {0, 1}, lambda in {0.5, 1, 2, 4}, seed 82, chosen by the validation primary metric. A tie within 0.5 points goes to the smaller k, then the smaller lambda. The best JEPA setting is carried to Stage 2 whether or not it beats LM-only on validation.
- Stage 2 seeds: 82, 23, 37 on the laptop; 82, 23, 37, 84, 4 on the cluster.
- Laptop datasets, in order: Spider, Synth, then Turk if time allows. All eight datasets and full fine-tuning on the cluster.

## 5. Arms
A1 LM-only (core). A2 static cosine-JEPA (core). A3 static JEPA with loss dropout at the paper's setting (core). A4 shuffled-pair JEPA with the best A2 settings (core, negative control). A0 zero-shot (optional; only with answer extraction). A5 linearity-only auxiliary loss (cluster, optional). A6 adaptive lambda (conditional, section 9). A7 InfoNCE with micro-batch of 8 or more (cluster, ablation).

## 6. Evaluation
- Greedy decoding with the defaults in `eval_spider.py` and `eval_exact.py` at tag `v2-prereg-2`; stop tokens read from the model's generation config.
- Synth and Turk: exact match after stripping whitespace (`eval_exact.py`).
- Spider: execution accuracy (`eval_spider.py`), reported over all rows, over rows whose gold query returns results, and with the share of predictions that fail to run. One test gold query fails to run; it stays in the denominator for every arm.
- GSM8K, Rotten Tomatoes, Yelp, NQ-Open, HellaSwag: scorers are added by amendment (section 11) with a gold-as-prediction check at 100% before any run that uses them.
- Spider prompts contain no schema, and every test database also appears in training; this is stated as a limitation.

## 7. Statistics
- Primary inference: paired bootstrap over test examples (10,000 resamples, metric averaged over seeds inside each resample), two-sided 95% interval, plus the per-seed differences. Exact McNemar tests on pooled predictions are descriptive.
- Holm correction across the two primary datasets for H0, and across all datasets for secondary results.
- Report effect sizes with intervals; report LoRA and full fine-tuning separately.

## 8. Fairness rule
Same train, validation and test files, prompts and chat template, decoding, scorer, seeds, optimizer settings and epochs for every arm and dataset-model pair, in the two stages above. JEPA arms cost extra compute; a matched-compute comparison is reported separately.

## 9. Controller (conditional)
The signal is the centered leave-one-out kernel-ridge residual on mid-layer mean-pooled embeddings, calibrated against an LM-only run. It must first separate real pairs from shuffled pairs, and tasks where JEPA helps from tasks where it does not (measured at windows of 16, 64 and 128 pairs). If it does not, the controller is not built and the diagnostic is reported instead.

## 10. Exclusions and reporting
Runs are excluded only for hash mismatch. Failed or diverged runs are reported, not dropped. Deviations are listed in the amendment log.

## 11. Amendment log (append only; date every entry; commit before the run it covers)
- 2026-11-04: tag `prereg-v1` was first created and pushed on commit `66177dc` (code only) before this file had been committed, because the file was not found when it was copied. That tag was deleted locally and on GitHub and re-created on the commit that contains this file. GitHub's history still shows the earlier tag push. No run of this protocol existed at that time.
- [ ] Qwen2-1.5B-Instruct revision hash.
- [ ] Llama-3.2-1B-Instruct revision hash, chat-template date handling, last-token offset, memory and speed.
- [ ] Stage 1 learning-rate results per dataset and model (outcomes, not changes to the protocol).
- [ ] Scorers for GSM8K, Rotten Tomatoes, Yelp, NQ-Open, HellaSwag.- 2026-11-04 (amendment 1): `Qwen/Qwen2-1.5B-Instruct` revision hash `ba1cf1846d7df0a0591d6c00649f57e798519da8` (Hugging Face cache at `D:\huggingface_cache\hub`).
