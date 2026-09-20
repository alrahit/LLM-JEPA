import json
import torch
import argparse
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def calculate_exact_match(prediction, ground_truth):
    return prediction.strip() == ground_truth.strip()

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Evaluate Exact Match Accuracy")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model folder")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the JSONL test file")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of dataset (e.g., spider, synth)")
    args = parser.parse_args()

    # Automatically format the output file name based on the dataset_name
    output_results_file = os.path.join("hybrid_adaptive", "results", f"{args.dataset_name}_eval_results.json")
    
    print(f"\nLoading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Load model in fp16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    print(f"Loading test data from {args.test_file}...")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    correct = 0
    total = len(lines)
    results = []

    print(f"Starting evaluation of {total} samples...")
    with torch.inference_mode():
        for line in tqdm(lines, total=total, desc="Evaluating"):
            data = json.loads(line)
            messages = data["messages"]
            
            prompt_messages = messages[:-1]
            ground_truth = messages[-1]["content"]

            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )

            generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            is_correct = calculate_exact_match(generated_text, ground_truth)
            if is_correct:
                correct += 1

            results.append({
                "prompt": prompt_messages[-1]["content"],
                "ground_truth": ground_truth,
                "generated_text": generated_text,
                "is_correct": is_correct
            })

    accuracy = (correct / total) * 100
    
    print("\n" + "="*50)
    print(f"🎉 FINAL EXACT MATCH ACCURACY ({args.dataset_name.upper()}): {accuracy:.2f}% ({correct}/{total})")
    print("="*50 + "\n")

    # Ensure results folder exists and save the dynamic file
    os.makedirs(os.path.dirname(output_results_file), exist_ok=True)
    with open(output_results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "dataset": args.dataset_name, 
            "accuracy": accuracy, 
            "total": total, 
            "correct": correct, 
            "details": results
        }, f, indent=2)
        
    print(f"Detailed failure/success log saved to: {output_results_file}")

if __name__ == "__main__":
    main()