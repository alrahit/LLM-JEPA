import json
import re
import os

def normalize_text(text):
    """Removes extra whitespace and lowers text for a fair structural comparison."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_structural_content(text, is_ground_truth=False):
    """Extracts raw code/data from markdown blocks or cleans conversational filler."""
    if not isinstance(text, str):
        return ""
        
    if is_ground_truth:
        return normalize_text(text)
        
    match = re.search(r'```[a-zA-Z]*\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return normalize_text(match.group(1))
    
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        if any(filler in line.lower() for filler in ["here is", "sure", "the query", "the output", "the answer"]):
            if ":" in line:
                line = line.split(":", 1)[-1]
            else:
                continue
        clean_lines.append(line)
        
    result = " ".join(clean_lines)
    return normalize_text(result)

def evaluate_dataset(file_path, dataset_name):
    if not os.path.exists(file_path):
        return
        
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
    
    for item in results_list:
        gt_text = item.get("ground_truth", "")
        pred_text = item.get("generated_text", item.get("predicted", ""))
        
        gt_clean = extract_structural_content(gt_text, is_ground_truth=True)
        pred_clean = extract_structural_content(pred_text, is_ground_truth=False)
        
        if gt_clean and pred_clean:
            total += 1
            if gt_clean == pred_clean or gt_clean in pred_clean:
                correct += 1
                
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"  -> Total Parsable Samples: {total}")
    print(f"  -> Correct Answers: {correct}")
    print(f"  -> True Accuracy: {accuracy:.2f}%\n")

if __name__ == "__main__":
    print("========================================")
    print("          SYNTH DATASET RESULTS         ")
    print("========================================")
    evaluate_dataset(r"hybrid_adaptive\results\synth_baseline_eval_results.json", "Baseline (Zero-Shot)")
    evaluate_dataset(r"hybrid_adaptive\results\synth_sft_eval_results.json", "Standard SFT Baseline")
    evaluate_dataset(r"hybrid_adaptive\results\synth_eval_results.json", "LinAlign Model (Ours)")
    
    print("========================================")
    print("         SPIDER DATASET RESULTS         ")
    print("========================================")
    evaluate_dataset(r"hybrid_adaptive\results\spider_baseline_eval_results.json", "Baseline (Zero-Shot)")
    evaluate_dataset(r"hybrid_adaptive\results\spider_sft_eval_results.json", "Standard SFT Baseline")
    evaluate_dataset(r"hybrid_adaptive\results\spider_eval_results.json", "LinAlign Model (Ours)")