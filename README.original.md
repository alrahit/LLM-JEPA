# LLM-JEPA-Research: Linear View-Alignment in Large Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

Welcome to the **LLM-JEPA Research Extension** repository. This project builds upon the Joint Embedding Predictive Architecture (JEPA) framework applied to Large Language Models, focusing on exploring linear view-alignment, representation spaces, and advanced fine-tuning methodologies.

---

## ?? Repository Structure

The workspace has been refactored into a modular, research-ready architecture:

```text
LLM-JEPA-Research
ÃÄÄ datasets/          # Research datasets (e.g., Spider data)
ÃÄÄ docs/              # Extended text guides and ancestral code usage
ÃÄÄ figures/           # Architecture diagrams, training curves, and plots
ÃÄÄ results/           # Raw JSONL evaluation logs and diagnostic readouts
ÃÄÄ scripts/           # Execution drivers (run.sh, setup.sh, run_stp.sh)
ÃÄÄ src/               # Main Python engine source code files
ÃÄÄ .gitignore         # Workspace configuration rules
ÃÄÄ CITATION.cff       # Academic citation metadata
ÀÄÄ README.md          # Project main information hub
```

---

## ?? Experimental Methodology & Results

This project houses custom fine-tuning execution data, testing baseline configurations against Lambda-scaled structural mutations:
* **Baseline Runs:** Documented inside `results/eval_results_regular.jsonl`.
* **JEPA Alignment Runs:** Tracked across `results/eval_results_jepa.jsonl`.
* **Lambda Variant Evaluations:** Structured inside `results/eval_results_lbd0.jsonl` and `results/eval_results_lbd1.jsonl`.

Data logs verify comparative variances in representation alignment metrics over targeted validation benchmarks.

---

## ?? Core Setup & Execution

### 1. Installation
Inspect configuration scripts prior to manual package alignment:
```bash
# Read the setup file to configure your local dependencies manually
cat scripts/setup.sh
```

### 2. Semantic Tube Prediction Fine-Tuning
Execute task runs by routing targeted flags to the engine module:
```bash
# Execute linear span tracking
python src/stp.py --linear=random_span --linear_predictor

# Run ablation variants
python src/stp.py --linear=e2e
```

### 3. LLM-JEPA Standard Fine-Tuning
```bash
# Run baseline optimization passes over 4 epochs
python src/finetune.py --additive_mask
```

---

## ?? References & Citation

The upstream architecture references core tasks detailed within standard joint embedding text-to-code alignments. If this extended implementation aids your experimentation, please reference the repository citation block via the interface metadata card or review details inside `CITATION.cff`.
