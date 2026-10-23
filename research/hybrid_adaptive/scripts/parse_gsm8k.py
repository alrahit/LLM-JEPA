import json
import re
import argparse

def extract_answer(text):
    """Extracts the final numerical answer from a block of text."""
    if not isinstance(text, str):
        return None

    # Clean commas for easier parsing (e.g., "65,000" -> "65000")
    text = text.replace(",", "")

    # If it has the standard delimiter, grab everything after it
    if "####" in text:
        ans = text.split("####")[-1].strip()
        # Try to grab the first valid number in that extracted chunk just in case there's trailing text
        match = re.search(r'-?\d+\.?\d*', ans)
        if match:
            return match.group(0)
        return ans

    # Fallback: grab the very last number in the entire text
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    return None

def evaluate_json_results(file_path):
    print(f"Parsing results from: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # The list of results is under the "details" key!
    if isinstance(data, dict) and "details" in data:
        results_list = data["details"]
    elif isinstance(data, list):
        results_list = data
    else:
        print("Error: Could not find the 'details' list in the file.")
        return

    correct = 0
    total = 0

    for item in results_list:
        gt_text = item.get("ground_truth", "")
        pred_text = item.get("generated_text", "") # Matches your JSON key exactly

        gt_num = extract_answer(gt_text)
        pred_num = extract_answer(pred_text)

        if gt_num is not None and pred_num is not None:
            total += 1
            try:
                # Convert to float to handle "3" vs "3.0" gracefully
                if float(gt_num) == float(pred_num):
                    correct += 1
            except ValueError:
                pass # Ignore if it failed to parse as a float

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"Total Parsable Samples: {total}")
    print(f"Correct Answers: {correct}")
    print(f"True Math Accuracy: {accuracy:.2f}%\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default=r"hybrid_adaptive\results\gsm8k_baseline_eval_results.json")
    parser.add_argument("--hybrid", type=str, default=r"hybrid_adaptive\results\gsm8k_eval_results.json")
    args = parser.parse_args()

    print("=== BASELINE (Raw Qwen2-1.5B) ===")
    evaluate_json_results(args.baseline)

    print("=== LINALIGN HYBRID (Ours) ===")
    evaluate_json_results(args.hybrid)