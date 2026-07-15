#!/bin/bash
#SBATCH --job-name=llm_jepa_linalign
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# ── log files (%j = SLURM job id) ──
#SBATCH --output=./log/linalign_%j.out
#SBATCH --error=./log/linalign_%j.err

set -euo pipefail

# ═══════════════════════════════════════════════════════════════
#  PATHS — EDIT THESE to match your setup
# ═══════════════════════════════════════════════════════════════
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"          # repo root (contains finetune.py, src/, datasets/)
LOG_DIR="${PROJECT_DIR}/log"
DATA_DIR="${PROJECT_DIR}/datasets"
RESULTS_DIR="${PROJECT_DIR}/results"
FIG_DIR="${PROJECT_DIR}/figures"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
SEED="${SEED:-42}"
NUM_EPOCHS="${NUM_EPOCHS:-4}"
LR="${LR:-1e-5}"
LORA_RANK="${LORA_RANK:-8}"
LAST_TOKEN="${LAST_TOKEN:--3}"

# ═══════════════════════════════════════════════════════════════
#  DATASET SELECTION — space-separated list of dataset prefixes.
#  Each must have datasets/<name>_train.jsonl and <name>_test.jsonl.
#  Per-dataset max_length is set in the loop below.
# ═══════════════════════════════════════════════════════════════
DATASETS="${DATASETS:-synth spider gsm8k}"

# ═══════════════════════════════════════════════════════════════
#  STAGE TOGGLES  (set to 1 to run, 0 to skip)
# ═══════════════════════════════════════════════════════════════
RUN_SANITY=1       # quick controller math test (python src/linalign.py)
RUN_CALIB=1        # Stage A: freeze lambda, MEASURE natural S per dataset
RUN_TRAIN=1        # Stage B: real LinAlign training (needs TAU from Stage A)
RUN_EVAL=1         # Stage C: evaluate the trained models
RUN_FIG=1          # Stage D: make lambda/S trace figures

# ═══════════════════════════════════════════════════════════════
#  TAU VALUES  (fill in AFTER running the calibration stage once)
#  Look at the "[read_tau] Suggested --linalign_tau ..." lines, then
#  set RUN_CALIB=0, paste the numbers here, and re-run with RUN_TRAIN=1.
#  Until then, leave as "AUTO" and the script uses the calibrated median.
# ═══════════════════════════════════════════════════════════════
declare -A TAU
TAU[synth]="AUTO"
TAU[spider]="AUTO"
TAU[gsm8k]="AUTO"
# e.g. after calibration:  TAU[synth]="0.0512"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}" "${FIG_DIR}"

RUN_LOG="${LOG_DIR}/linalign_run_${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S).log"

# ═══════════════════════════════════════════════════════════════
#  ENVIRONMENT — EDIT to match your cluster
# ═══════════════════════════════════════════════════════════════
# If using conda:
# source /path/to/miniconda3/etc/profile.d/conda.sh
# conda activate jepa || { echo "conda activate failed"; exit 1; }
#
# If using a venv:
# source "${PROJECT_DIR}/venv/bin/activate"

cd "${PROJECT_DIR}"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-${PROJECT_DIR}/_hf_cache}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}"

# max_length per dataset (short tasks -> small; long solutions -> big)
maxlen_for() {
  case "$1" in
    gsm8k)   echo 512 ;;
    spider)  echo 256 ;;
    nq_open) echo 256 ;;
    *)       echo 128 ;;
  esac
}

# ═══════════════════════════════════════════════════════════════
#  RUN  (tee everything to RUN_LOG)
# ═══════════════════════════════════════════════════════════════
{
  echo "════════════════════════════════════════════════════════════"
  echo "  LLM-JEPA + LinAlign  —  run log"
  echo "════════════════════════════════════════════════════════════"
  echo "  Date        : $(date)"
  echo "  Host        : $(hostname)"
  echo "  Job ID      : ${SLURM_JOB_ID:-N/A}"
  echo "  Project     : ${PROJECT_DIR}"
  echo "  Model       : ${MODEL_NAME}"
  echo "  Datasets    : ${DATASETS}"
  echo "  Seed        : ${SEED}   Epochs: ${NUM_EPOCHS}   LR: ${LR}   LoRA r: ${LORA_RANK}"
  echo "  Stages      : sanity=${RUN_SANITY} calib=${RUN_CALIB} train=${RUN_TRAIN} eval=${RUN_EVAL} fig=${RUN_FIG}"
  echo "  CUDA devices: ${CUDA_VISIBLE_DEVICES}"
  echo "  Python      : $(which python)  ($(python --version 2>&1))"
  echo "------------------------------------------------------------"
  nvidia-smi || echo "  (nvidia-smi unavailable)"
  python -c "import torch; print('   torch', torch.__version__, '| cuda?', torch.cuda.is_available())"
  echo "════════════════════════════════════════════════════════════"
  echo

  # ── sanity: files present ──
  if [[ ! -f "${PROJECT_DIR}/finetune.py" ]]; then
    echo "ERROR: finetune.py not found in ${PROJECT_DIR}"; exit 1
  fi
  if [[ ! -f "${PROJECT_DIR}/src/linalign.py" ]]; then
    echo "ERROR: src/linalign.py not found. Create it first."; exit 1
  fi

  # ════════════════════════════════════════════════════════════
  #  STAGE: SANITY — controller math (no GPU/model needed)
  # ════════════════════════════════════════════════════════════
  if [[ "${RUN_SANITY}" == "1" ]]; then
    echo "############################################################"
    echo "#  SANITY : python src/linalign.py"
    echo "############################################################"
    python src/linalign.py
    echo "  sanity done at $(date)"; echo
  else
    echo "  [skip] SANITY"; echo
  fi

  # ════════════════════════════════════════════════════════════
  #  STAGE A: CALIBRATION — measure natural S (lambda FROZEN, eta=0)
  # ════════════════════════════════════════════════════════════
  if [[ "${RUN_CALIB}" == "1" ]]; then
    echo "############################################################"
    echo "#  STAGE A : tau calibration (freeze lambda, measure S)"
    echo "############################################################"
    for DS in ${DATASETS}; do
      TRAIN_FILE="${DATA_DIR}/${DS}_train.jsonl"
      ML=$(maxlen_for "${DS}")
      CALIB_LOG="${RESULTS_DIR}/calib_${DS}.json"
      if [[ ! -f "${TRAIN_FILE}" ]]; then
        echo "  [skip] ${DS}: ${TRAIN_FILE} not found"; continue
      fi
      echo "------------------------------------------------------------"
      echo "  Calibrating: ${DS}  (max_length=${ML})"
      echo "------------------------------------------------------------"
      python finetune.py \
        --train_file "${TRAIN_FILE}" \
        --output_dir "./calib-tmp-${DS}" \
        --num_epochs 1 --finetune_seed "${SEED}" --model_name "${MODEL_NAME}" \
        --learning_rate "${LR}" --lora --lora_rank "${LORA_RANK}" \
        --last_token "${LAST_TOKEN}" --predictors 0 --max_length "${ML}" \
        --linalign --linalign_eta 0.0 --linalign_tau 999 \
        --linalign_min_batch 16 --linalign_log "${CALIB_LOG}"
      rm -rf "./calib-tmp-${DS}" 2>/dev/null || true
      echo "  -- suggested tau for ${DS}:"
      python src/read_tau.py "${CALIB_LOG}" | sed 's/^/     /'
      echo
    done
    echo "  STAGE A done at $(date)"
    echo "  >>> Copy the suggested tau values into the TAU[] map above,"
    echo "  >>> then set RUN_CALIB=0 and RUN_TRAIN=1 and re-run."
    echo
  else
    echo "  [skip] STAGE A (calibration)"; echo
  fi

  # helper: resolve tau (use TAU[] if set, else median from calib log)
  resolve_tau() {
    local ds="$1"
    local t="${TAU[$ds]:-AUTO}"
    if [[ "${t}" != "AUTO" ]]; then
      echo "${t}"; return
    fi
    local clog="${RESULTS_DIR}/calib_${ds}.json"
    if [[ -f "${clog}" ]]; then
      python src/read_tau.py "${clog}" 2>/dev/null | awk '/Suggested/{print $3}'
    else
      echo "0.05"   # last-resort default
    fi
  }

  # ════════════════════════════════════════════════════════════
  #  STAGE B: TRAIN — real LinAlign runs
  # ════════════════════════════════════════════════════════════
  if [[ "${RUN_TRAIN}" == "1" ]]; then
    echo "############################################################"
    echo "#  STAGE B : LinAlign training"
    echo "############################################################"
    for DS in ${DATASETS}; do
      TRAIN_FILE="${DATA_DIR}/${DS}_train.jsonl"
      ML=$(maxlen_for "${DS}")
      TAU_VAL=$(resolve_tau "${DS}")
      OUT_DIR="./model-linalign-${DS}"
      HIST="${RESULTS_DIR}/linalign_${DS}_seed${SEED}.json"
      if [[ ! -f "${TRAIN_FILE}" ]]; then
        echo "  [skip] ${DS}: ${TRAIN_FILE} not found"; continue
      fi
      echo "------------------------------------------------------------"
      echo "  Training: ${DS}  (max_length=${ML}, tau=${TAU_VAL})"
      echo "------------------------------------------------------------"
      python finetune.py \
        --train_file "${TRAIN_FILE}" \
        --output_dir "${OUT_DIR}" \
        --num_epochs "${NUM_EPOCHS}" --finetune_seed "${SEED}" --model_name "${MODEL_NAME}" \
        --learning_rate "${LR}" --lora --lora_rank "${LORA_RANK}" \
        --last_token "${LAST_TOKEN}" --predictors 0 --max_length "${ML}" \
        --linalign --linalign_lambda_init 1.0 --linalign_eta 0.5 \
        --linalign_tau "${TAU_VAL}" --linalign_min_batch 16 \
        --linalign_log "${HIST}"
      echo "  ${DS} training done at $(date)"; echo
    done
    echo "  STAGE B done at $(date)"; echo
  else
    echo "  [skip] STAGE B (training)"; echo
  fi

  # ════════════════════════════════════════════════════════════
  #  STAGE C: EVALUATE
  # ════════════════════════════════════════════════════════════
  if [[ "${RUN_EVAL}" == "1" ]]; then
    echo "############################################################"
    echo "#  STAGE C : evaluation"
    echo "############################################################"
    for DS in ${DATASETS}; do
      TEST_FILE="${DATA_DIR}/${DS}_test.jsonl"
      OUT_DIR="./model-linalign-${DS}"
      EVAL_OUT="${RESULTS_DIR}/eval_linalign_${DS}.jsonl"
      if [[ ! -d "${OUT_DIR}" ]]; then
        echo "  [skip] ${DS}: model ${OUT_DIR} not found (train first)"; continue
      fi
      if [[ ! -f "${TEST_FILE}" ]]; then
        echo "  [skip] ${DS}: ${TEST_FILE} not found"; continue
      fi
      echo "------------------------------------------------------------"
      echo "  Evaluating: ${DS}"
      echo "------------------------------------------------------------"
      python evaluate.py \
        --model_name "${OUT_DIR}" \
        --input_file "${TEST_FILE}" \
        --output_file "${EVAL_OUT}" \
        --original_model_name "${MODEL_NAME}" \
        --nosplit_data --no_skip_existing --split_tune_untune
      echo "  ${DS} eval done at $(date)"; echo
    done
    echo "  STAGE C done at $(date)"; echo
  else
    echo "  [skip] STAGE C (evaluation)"; echo
  fi

  # ════════════════════════════════════════════════════════════
  #  STAGE D: FIGURES
  # ════════════════════════════════════════════════════════════
  if [[ "${RUN_FIG}" == "1" ]]; then
    echo "############################################################"
    echo "#  STAGE D : figures"
    echo "############################################################"
    for DS in ${DATASETS}; do
      HIST="${RESULTS_DIR}/linalign_${DS}_seed${SEED}.json"
      if [[ ! -f "${HIST}" ]]; then
        echo "  [skip] ${DS}: history ${HIST} not found"; continue
      fi
      echo "  Plotting controller dynamics for ${DS}"
      python src/analyze_linalign.py --history "${HIST}" --outdir "${FIG_DIR}"
    done
    echo "  STAGE D done at $(date)"; echo
  else
    echo "  [skip] STAGE D (figures)"; echo
  fi

  echo "════════════════════════════════════════════════════════════"
  echo "  ALL DONE at $(date)"
  echo "════════════════════════════════════════════════════════════"

} 2>&1 | tee "${RUN_LOG}"

echo "Full run log saved to: ${RUN_LOG}"
