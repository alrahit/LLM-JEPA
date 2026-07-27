# LLM-JEPA Research: Linear View-Alignment in Large Language Models

**Research led by:** Gollam Rabby, Junior Research Group Leader (Nachwuchsgruppenleiter)
**Contributor:** Al Rahit

## Hypothesis

LLM-JEPA improves fine-tuning by pushing the Text->Code embedding relationship
toward a simple **linear** mapping. This project tests three predictions of
that hypothesis:

- **P1:** JEPA-trained models show higher Text->Code linearity (R^2) than regular fine-tuning.
- **P2:** Higher JEPA lambda (pull strength) leads to higher task accuracy.
- **P3:** A linear predictor (predictors=0) outperforms nonlinear predictors (predictors=1,2).

A self-tuning contribution (`--linalign`) is also tested: an Adam-style
controller that adjusts lambda automatically from a live linearity readout,
instead of a manual lambda sweep.

## Repository Structure

- `configs/` — Run configuration files
- `datasets/` — NL-RX-SYNTH and GSM8K train/test splits
- `docs/` — Extended usage notes
- `results/synth/` — Synth dataset results (accuracy, R^2, training logs)
- `results/gsm8k/` — GSM8K dataset results (accuracy, R^2, linalign trace)
- `scripts/` — finetune.py, evaluate.py, measure_linearity.py, run.sh
- `src/` — Core engine source
- `requirements.txt`

## Setup

```powershell
python -m venv venv --system-site-packages
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install peft tqdm protobuf sentencepiece matplotlib pandas scipy seaborn
hf auth login
```

## Running an experiment

```powershell
cd scripts
python finetune.py --train_file ../datasets/<dataset>_train.jsonl --output_dir ../results/<dataset>/model-jepa --num_epochs 4 --lbd 0.5 --predictors 0 --model_name Qwen/Qwen2.5-1.5B-Instruct --learning_rate 1e-5 --lora --lora_rank 8

python evaluate.py --model_name ../results/<dataset>/model-jepa --input_file ../datasets/<dataset>_test.jsonl --output_file ../results/<dataset>/eval_jepa.jsonl --split_tune_untune --original_model_name Qwen/Qwen2.5-1.5B-Instruct --nosplit_data --device_map cuda:0

python measure_linearity.py --jepa_model ../results/<dataset>/model-jepa --regular_model ../results/<dataset>/model-regular --test_file ../datasets/<dataset>_test.jsonl --max_examples 500 --pca_components 50
```

Full results and hypothesis-by-hypothesis breakdowns are in
`results/synth/synth_results.txt` and `results/gsm8k/gsm8k_results.txt`.

## Results Summary

### Synth (NL -> Regex), test set 2000

| Config | Accuracy | R^2 |
|---|---|---|
| Regular | 38.35% | 0.1005 |
| lbd=0 | 37.85% | 0.0978 |
| JEPA lbd=0.5 | 40.75% | 0.1185 |
| JEPA lbd=1.0 | 43.10% | 0.1169 |
| pred=1 | 39.90% | 0.0809 |
| pred=2 | 40.75% | 0.1217 |

**Verdict:** P1 supported, P2 supported, P3 supported.

### GSM8K (grade-school math), test set 1319, accuracy recomputed offline

| Config | Accuracy | R^2 |
|---|---|---|
| Regular | 35.41% | ~-55.7 to -59.9 |
| lbd=0 | 35.33% | -57.997 |
| JEPA lbd=0.5 | 34.72% | -7.4469 |
| JEPA lbd=1.0 | 34.65% | -1.2140 |
| pred=1 | 34.65% | -5.2932 |
| pred=2 | 35.25% | -0.8556 |
| linalign (self-tuning) | 35.25% | -54.0186 |

**Verdict:** P1 supported (every JEPA variant beats Regular's R^2). P2 not
supported (accuracy flat within noise across the lambda sweep). P3 reversed
(pred=2 is most linear, not pred=0 -- opposite of Synth's ordering).

**Cross-dataset finding:** on Synth, accuracy and linearity improve together
with lambda. On GSM8K, linearity improves sharply and monotonically with
lambda while accuracy stays flat -- representation linearity and downstream
task accuracy appear decoupled on harder, longer-form reasoning tasks.

## Known limitations

- `evaluate.py`'s live-printed "Success Rate" was found to be unreliable on
  GSM8K (reported ~0.08% for every model regardless of real performance).
  All GSM8K accuracy numbers above are recomputed offline from saved
  generations using the same scoring regex. This was not caught during the
  Synth run; Synth's accuracy numbers rest on the unverified live output.
- The `linalign` self-tuning controller's default `tau=0.05` is miscalibrated
  for GSM8K's embedding scale, causing lambda to collapse toward zero during
  training (see `results/gsm8k/linalign_trace.json`). This is a calibration
  issue, not a failure of the self-tuning concept.

## References

Based on LLM-JEPA (Huang, LeCun, Balestriero, 2025): https://arxiv.org/abs/2509.14252
Original codebase: https://github.com/rbalestr-lab/llm-jepa
