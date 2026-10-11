# PRE-REGISTRATION DRAFT (finalize BEFORE any new training; do not edit after seeing results)

## Question
Does an adaptive-lambda JEPA controller match the best static lambda without a per-task sweep,
and avoid harm where alignment is trivial or ill-posed?

## Fixed setup
- Model Qwen/Qwen2-1.5B-Instruct, LoRA r=8 alpha=16, lr 1e-5, 4 epochs, effective batch 16, seeds 42/43/44.
- Frozen data (sha256 in dataset_hashes_2026-10-09.txt). Any tuning uses a validation split carved from train; test is used once per final run.
- Eval: evaluate_synth.py (greedy, 256 new tokens). Primary metric: parsed accuracy over ALL test samples. Secondary: raw EM; SPIDER execution accuracy (to be added, decided now, applied to all arms).

## Arms
A0 zero-shot; A1 LM-only (same code path, lambda=0); A2 static cosine-JEPA lambda in {0.1, 1.0};
A3 static InfoNCE lambda in {0.1, 1.0}; A4 adaptive lambda. Rotten Tomatoes: cosine arms only (InfoNCE is ill-posed there).

## Hypotheses (margins TO BE CONFIRMED before running)
H1: on SYNTH the best static arm beats A1 by >= 1.0 pt (paired, mean over seeds). If H1 fails: report that JEPA gives no measurable benefit here and do NOT claim a LinAlign improvement.
H2: A4 is within 1.0 pt of the best static arm on every dataset (non-inferiority), without a sweep.
H3: on SPIDER and GSM8K lambda in A4 falls below 0.1 x lambda_init within the first 10% of training and accuracy is not worse than A1 by > 1.0 pt.

## Rules
- tau, eta are chosen once on a SYNTH validation pilot and then frozen for all datasets.
- All runs are reported, including failed ones. No test-set tuning. No metric changes after results.
- Known limitation to state: SPIDER/GSM8K pairing is solvable by lexical overlap (lexical_baseline.py).
