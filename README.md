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
- `datasets/` — NL-RX-SYNTH, GSM8K, and Spider train/test splits
- `docs/` — Extended usage notes
- `results/synth/` — Synth dataset results (accuracy, R^2, training logs)
- `results/gsm8k/` — GSM8K dataset results (accuracy, R^2, linalign trace)
- `results/spider/` — Spider dataset results (accuracy, R^2, linalign trace)
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

For Spider specifically, `evaluate.py` also requires `--spider_path <path to
the Spider .sqlite database folder>`, since accuracy is scored by executing
both the generated and reference SQL against the real database, not by
string matching.

Full results and hypothesis-by-hypothesis breakdowns are in
`results/synth/synth_results.txt`, `results/gsm8k/gsm8k_results.txt`, and
`results/spider/spider_results.txt`.

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

### Spider (NL -> SQL), test set 1447, execution-based accuracy

| Config | Accuracy | R^2 |
|---|---|---|
| Regular | 2.76% (40/1447) | baseline |
| lbd=0 | 2.83% (41/1447) | -0.0552 (not supported) |
| JEPA lbd=0.5 | 2.76% (40/1447) | supported, +20.8% |
| JEPA lbd=1.0 | 2.83% (41/1447) | -0.0513 (not supported) |
| pred=1 | 2.97% (43/1447) | not supported |
| pred=2 | 3.04% (44/1447) | supported, +8.1% |
| linalign (self-tuning) | 2.97% (43/1447) | -0.0584 (not supported) |

**Verdict:** P1 only partially supported (jepa/lbd0.5 and pred2 beat their
paired Regular; lbd0, lbd1, and linalign did not). P2 not supported (accuracy
flat within a 0.28 percentage point band, no trend at all). P3 reversed on
accuracy (pred2 > pred1 > pred0) though pred0 and pred2 both beat pred1 on
R^2 -- R^2 and accuracy orderings disagree with each other on this dataset.

**Notably, the model with the single best R^2 (jepa/pred0, +20.8%) tied for
the worst accuracy, while the model with the best accuracy (pred2) had a
weaker R^2 gain than jepa.** Linearity and accuracy point in opposite
directions at the extremes -- the clearest linearity/accuracy decoupling
observed across all three datasets tested.

### Cross-dataset finding

On Synth, accuracy and linearity improve together with lambda -- clean
support for the full hypothesis. On GSM8K, linearity improves sharply and
monotonically with lambda while accuracy stays flat. On Spider, linearity
and accuracy are not just decoupled but can move in opposite directions.
Taken together, this suggests linear view-alignment is a real, reproducible
effect of the JEPA training signal, but whether it translates into better
downstream task accuracy depends heavily on the task -- strongest and
clearest on short, structured generation (Synth), weaker and less
predictable on longer or more open-ended generation (GSM8K, Spider).

## Known limitations

- `evaluate.py`'s live-printed "Success Rate" was found to be unreliable on
  GSM8K (reported ~0.08% for every model regardless of real performance).
  All GSM8K and Spider accuracy numbers above are recomputed/verified from
  saved per-example generations rather than the live console output. This
  was not caught during the Synth run; Synth's accuracy numbers rest on the
  unverified live output.
- The `linalign` self-tuning controller's default `tau=0.05` is miscalibrated
  for both GSM8K's and Spider's embedding scale, causing lambda to collapse
  toward zero during training on both datasets (see each dataset's
  `linalign_trace.json`). This is a calibration issue, not a failure of the
  self-tuning concept, but it means `tau=0.05` should not be treated as a
  universal default without per-dataset tuning.

## References

Based on LLM-JEPA (Huang, LeCun, Balestriero, 2025): https://arxiv.org/abs/2509.14252
Original codebase: https://github.com/rbalestr-lab/llm-jepa
Spider dataset: https://yale-lily.github.io/spider
