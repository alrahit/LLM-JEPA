# run_spider_pipeline.ps1
# Resumable Spider dataset pipeline for LLM-JEPA research.
# Safe to stop and re-run at any time -- already-completed phases are
# skipped automatically. Only keeps model-regular's checkpoint (needed
# as the reference for every linearity comparison); other phases'
# checkpoints are deleted right after their eval+linearity are saved,
# to keep disk usage low.

$ErrorActionPreference = "Stop"

# ---- EDIT THIS if you have the Spider .sqlite database files ----
# Leave as "" to train-only and skip evaluation until you set this.
$spiderPath = ""
# -------------------------------------------------------------------

Set-Location D:\JEPA\LLM-JEPA\scripts
$env:HF_HOME = "D:\huggingface_cache"

$model  = "Qwen/Qwen2.5-1.5B-Instruct"
$lr     = "1e-5"
$seed   = 42
$out    = "..\results\spider"
$train  = "..\datasets\spider_train.jsonl"
$test   = "..\datasets\spider_test.jsonl"

New-Item -ItemType Directory -Force -Path $out | Out-Null

function Test-ModelSaved($dir) {
    if (-not (Test-Path $dir)) { return $false }
    $hasWeights = (Get-ChildItem $dir -Filter "*.safetensors" -ErrorAction SilentlyContinue).Count -gt 0
    $hasWeights = $hasWeights -or (Get-ChildItem $dir -Filter "pytorch_model*.bin" -ErrorAction SilentlyContinue).Count -gt 0
    return $hasWeights
}

function Run-Phase {
    param(
        [string]$name,
        [string[]]$extraArgs,
        [switch]$keepModel
    )

    $modelDir = "$out\model-$name"
    $evalFile = "$out\eval_$name.jsonl"
    $linFile  = "$out\linearity_results_$name.json"

    $evalDone = (Test-Path $evalFile) -and ((Get-Item $evalFile).Length -gt 0)
    $linDone  = Test-Path $linFile

    if ($evalDone -and $linDone) {
        Write-Host "=== [$name] already fully complete -- skipping ===" -ForegroundColor Green
        return
    }

    if (-not (Test-ModelSaved $modelDir)) {
        Write-Host "=== [$name] training ===" -ForegroundColor Cyan
        if (Test-Path $modelDir) { Remove-Item -Recurse -Force $modelDir }
        $trainArgs = @(
            "--train_file", $train, "--output_dir", $modelDir,
            "--num_epochs", "4", "--finetune_seed", $seed, "--max_length", "128",
            "--model_name", $model, "--learning_rate", $lr, "--lora", "--lora_rank", "8"
        ) + $extraArgs
        python finetune.py @trainArgs
    } else {
        Write-Host "=== [$name] model already trained -- skipping training ===" -ForegroundColor Yellow
    }

    if (-not $evalDone) {
        if ($spiderPath -eq "") {
            Write-Host "=== [$name] SKIPPING EVAL -- spiderPath not set at top of script ===" -ForegroundColor Red
            return
        }
        Write-Host "=== [$name] evaluating ===" -ForegroundColor Cyan
        python evaluate.py --model_name $modelDir --input_file $test --output_file $evalFile `
            --split_tune_untune --original_model_name $model --nosplit_data --device_map "cuda:0" `
            --spider_path $spiderPath
    }

    if (-not $linDone) {
        Write-Host "=== [$name] measuring linearity ===" -ForegroundColor Cyan
        python measure_linearity.py --jepa_model $modelDir --regular_model "$out\model-regular" `
            --test_file $test --max_examples 500 --pca_components 50
        Move-Item -Force linearity_results.json $linFile
    }

    if (-not $keepModel -and (Test-Path $modelDir)) {
        Write-Host "=== [$name] freeing disk space (deleting model checkpoint) ===" -ForegroundColor Magenta
        Remove-Item -Recurse -Force $modelDir
    }
}

Run-Phase -name "regular"  -extraArgs @("--regular") -keepModel
Run-Phase -name "lbd0"     -extraArgs @("--last_token","-3","--lbd","0.0","--predictors","0")
Run-Phase -name "jepa"     -extraArgs @("--last_token","-3","--lbd","0.5","--predictors","0")
Run-Phase -name "lbd1"     -extraArgs @("--last_token","-3","--lbd","1.0","--predictors","0")
Run-Phase -name "pred1"    -extraArgs @("--last_token","-3","--lbd","0.5","--predictors","1")
Run-Phase -name "pred2"    -extraArgs @("--last_token","-3","--lbd","0.5","--predictors","2")
Run-Phase -name "linalign" -extraArgs @("--last_token","-3","--predictors","0","--linalign","--linalign_tau","0.05","--linalign_log","$out\linalign_trace.json")

Write-Host "=== SPIDER PIPELINE RUN COMPLETE (or stopped -- re-run this script to resume) ===" -ForegroundColor Green
