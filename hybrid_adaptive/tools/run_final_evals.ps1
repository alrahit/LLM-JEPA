$env:PYTHONUTF8="1"; $env:PYTHONIOENCODING="utf-8"
$C    = "hybrid_adaptive\results\qwen\checkpoints"
$BASE = "Qwen/Qwen2-1.5B-Instruct"

$jobs = @(
  @("final_synth_linalign",  "$C\arm1_hybrid_retrieval",    "datasets\synth_test.jsonl"),
  @("final_spider_linalign", "$C\spider_hybrid_retrieval",  "datasets\spider_test.jsonl"),
  @("final_synth_sft",       "$C\qwen_sft_baseline",        "datasets\synth_test.jsonl"),
  @("final_synth_zeroshot",  $BASE,                         "datasets\synth_test.jsonl"),
  @("final_spider_zeroshot", $BASE,                         "datasets\spider_test.jsonl")
)

foreach ($j in $jobs) {
  $out = "hybrid_adaptive\results\$($j[0])_eval_results.json"
  if (Test-Path $out) { "SKIP $($j[0]) (already exists)"; continue }
  "=== $(Get-Date)  START $($j[0])"
  python hybrid_adaptive\scripts\evaluate_synth.py --model_path $j[1] --test_file $j[2] --dataset_name $j[0]
  if (Test-Path $out) {
    $r = Get-Content $out -Raw | ConvertFrom-Json
    "RESULT $($j[0]): acc=$($r.accuracy) correct=$($r.correct)/$($r.total)"
  } else { "FAILED $($j[0])" }
  "=== $(Get-Date)  END $($j[0])"
}
