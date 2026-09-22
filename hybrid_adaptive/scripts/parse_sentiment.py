import json
import re

def extract_sentiment(text):
    """Finds the first occurrence of 'good' or 'bad' in the text."""
    if not isinstance(text, str):
        return None
    
    text = text.lower()
    # Search for the exact word "good" or "bad"
    match = re.search(r'\b(good|bad)\b', text)
    if match:
        return match.group(1)
    return None

def evaluate_sentiment(file_path, dataset_name):
    print(f"Parsing {dataset_name} results from: {file_path}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  -> Error: Could not read file. {e}\n")
        return

    if isinstance(data, dict) and "details" in data:
        results_list = data["details"]
    elif isinstance(data, list):
        results_list = data
    else:
        print("  -> Error: Could not find the 'details' list.\n")
        return

    correct = 0
    total = 0
    parsable = 0
    
    for item in results_list:
        gt_text = str(item.get("ground_truth", "")).lower().strip()
        pred_text = item.get("generated_text", item.get("predicted", ""))
        
        pred_sentiment = extract_sentiment(pred_text)
        
        total += 1
        if pred_sentiment:
            parsable += 1
            if pred_sentiment == gt_text:
                correct += 1
                
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"  -> Total Samples: {total}")
    print(f"  -> Parsable Outputs (Found good/bad): {parsable}")
    print(f"  -> Correct Answers: {correct}")
    print(f"  -> True Sentiment Accuracy: {accuracy:.2f}%\n")

if __name__ == "__main__":
    print("========================================")
    print("      ROTTEN TOMATOES DATASET RESULTS   ")
    print("========================================")
    evaluate_sentiment(r"hybrid_adaptive\results\rotten_tomatoes_baseline_eval_results.json", "Baseline (Raw Qwen2-1.5B)")
    evaluate_sentiment(r"hybrid_adaptive\results\rotten_tomatoes_eval_results.json", "LinAlign Model (Ours)")