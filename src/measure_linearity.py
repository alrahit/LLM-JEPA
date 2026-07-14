"""
R² Linearity Measurement Script for LLM-JEPA Hypothesis
=========================================================
Hypothesis: JEPA works because it pushes Text->Code relationship to be LINEAR.

This script:
1. Loads both JEPA and Regular fine-tuned models
2. Extracts Text embeddings and Code embeddings from test examples
3. Reduces dimensions using PCA (fixes curse of dimensionality)
4. Fits a LINEAR regression: Text_embedding -> Code_embedding
5. Measures R² score for both models
6. Higher R² = more linear Text->Code relationship

If JEPA R² > Regular R² → Hypothesis SUPPORTED ✅
"""

import json
import torch
import numpy as np
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import argparse

# ─────────────────────────────────────────────
# 1. Helper Functions
# ─────────────────────────────────────────────

def get_user_messages(messages):
    return copy.deepcopy(messages)[1:2]

def get_assistant_messages(messages):
    return messages[2:3]

def format_conversation(messages, tokenizer):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

def get_embedding(model, tokenizer, prompt, max_length=128, layer=-1):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[layer]
    embedding = hidden_states[0, -1, :].float().cpu().numpy()
    return embedding

def load_model(model_path):
    print(f"\nLoading: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"Model loaded on CPU")
    return model, tokenizer

# ─────────────────────────────────────────────
# 2. Extract Embeddings
# ─────────────────────────────────────────────

def extract_embeddings(model, tokenizer, dataset, max_examples=500):
    text_embeddings = []
    code_embeddings = []
    print(f"Extracting embeddings from {max_examples} examples...")
    for i, example in enumerate(dataset[:max_examples]):
        if i % 50 == 0:
            print(f"  Progress: {i}/{max_examples}")
        messages = example['messages']
        try:
            text_msgs = get_user_messages(messages)
            text_prompt = format_conversation(text_msgs, tokenizer)
            text_emb = get_embedding(model, tokenizer, text_prompt)

            code_msgs = get_assistant_messages(messages)
            code_prompt = format_conversation(code_msgs, tokenizer)
            code_emb = get_embedding(model, tokenizer, code_prompt)

            text_embeddings.append(text_emb)
            code_embeddings.append(code_emb)
        except Exception as e:
            print(f"  Skipping example {i}: {e}")
            continue

    text_embeddings = np.array(text_embeddings)
    code_embeddings = np.array(code_embeddings)
    print(f"Extracted {len(text_embeddings)} embedding pairs")
    print(f"Embedding shape: {text_embeddings.shape}")
    return text_embeddings, code_embeddings

# ─────────────────────────────────────────────
# 3. Measure R² with PCA
# ─────────────────────────────────────────────

def measure_linearity(text_embeddings, code_embeddings, model_name, n_components=50):
    """
    Fit linear regression with PCA dimensionality reduction.
    PCA reduces 1536 dims -> 50 dims to avoid overfitting.
    """
    print(f"\nMeasuring linearity for: {model_name}")

    # Normalize
    scaler_text = StandardScaler()
    scaler_code = StandardScaler()
    text_norm = scaler_text.fit_transform(text_embeddings)
    code_norm = scaler_code.fit_transform(code_embeddings)

    # PCA dimensionality reduction
    n_components = min(n_components, len(text_norm) - 1, text_norm.shape[1])
    pca_text = PCA(n_components=n_components)
    pca_code = PCA(n_components=n_components)
    text_reduced = pca_text.fit_transform(text_norm)
    code_reduced = pca_code.fit_transform(code_norm)

    print(f"  Reduced dimensions: {text_embeddings.shape[1]} -> {n_components}")
    print(f"  Text PCA variance explained: {pca_text.explained_variance_ratio_.sum():.3f}")
    print(f"  Code PCA variance explained: {pca_code.explained_variance_ratio_.sum():.3f}")

    # Split 80/20
    n = len(text_reduced)
    split = int(0.8 * n)
    X_train, X_test = text_reduced[:split], text_reduced[split:]
    y_train, y_test = code_reduced[:split], code_reduced[split:]

    # Fit linear regression
    reg = Ridge(alpha=1.0)
    reg.fit(X_train, y_train)

    # Measure R²
    y_pred = reg.predict(X_test)
    r2 = r2_score(y_test, y_pred, multioutput='uniform_average')

    print(f"  R² Score: {r2:.4f}")
    print(f"  Interpretation: {interpret_r2(r2)}")
    return r2

def interpret_r2(r2):
    if r2 > 0.8:
        return "Very strong linear relationship ✅✅✅"
    elif r2 > 0.6:
        return "Strong linear relationship ✅✅"
    elif r2 > 0.4:
        return "Moderate linear relationship ✅"
    elif r2 > 0.2:
        return "Weak linear relationship ⚠️"
    elif r2 > 0.0:
        return "Very weak linear relationship ⚠️⚠️"
    else:
        return "Non-linear relationship ❌"

# ─────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa_model", type=str, default="./fine-tuned")
    parser.add_argument("--regular_model", type=str, default="./model-regular")
    parser.add_argument("--test_file", type=str, default="datasets/synth_test.jsonl")
    parser.add_argument("--max_examples", type=int, default=500)
    parser.add_argument("--pca_components", type=int, default=50)
    args = parser.parse_args()

    print("=" * 60)
    print("LLM-JEPA Linearity Measurement (R² Score + PCA)")
    print("Hypothesis: JEPA R² > Regular R²")
    print("=" * 60)

    # Load dataset
    print(f"\nLoading dataset: {args.test_file}")
    with open(args.test_file, 'r') as f:
        dataset = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(dataset)} examples, using {args.max_examples}")

    results = {}

    # JEPA Model
    jepa_model, jepa_tokenizer = load_model(args.jepa_model)
    jepa_text_emb, jepa_code_emb = extract_embeddings(jepa_model, jepa_tokenizer, dataset, args.max_examples)
    jepa_r2 = measure_linearity(jepa_text_emb, jepa_code_emb, "JEPA", args.pca_components)
    results['jepa'] = jepa_r2
    del jepa_model, jepa_tokenizer

    # Regular Model
    reg_model, reg_tokenizer = load_model(args.regular_model)
    reg_text_emb, reg_code_emb = extract_embeddings(reg_model, reg_tokenizer, dataset, args.max_examples)
    reg_r2 = measure_linearity(reg_text_emb, reg_code_emb, "Regular", args.pca_components)
    results['regular'] = reg_r2
    del reg_model, reg_tokenizer

    # Final Results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  JEPA    R²: {results['jepa']:.4f}  {interpret_r2(results['jepa'])}")
    print(f"  Regular R²: {results['regular']:.4f}  {interpret_r2(results['regular'])}")
    print(f"  Difference: {results['jepa'] - results['regular']:+.4f}")
    print()

    if results['jepa'] > results['regular']:
        diff_pct = abs(results['jepa'] - results['regular']) / (abs(results['regular']) + 1e-9) * 100
        print("✅ HYPOTHESIS SUPPORTED!")
        print(f"   JEPA creates MORE LINEAR Text->Code representations.")
        print(f"   JEPA R² is {diff_pct:.1f}% higher than Regular.")
    else:
        print("❌ HYPOTHESIS NOT SUPPORTED")
        print("   Regular has equal or more linear representations than JEPA.")

    print("=" * 60)

    # Save results
    with open('linearity_results.json', 'w') as f:
        json.dump({
            'jepa_r2': results['jepa'],
            'regular_r2': results['regular'],
            'difference': results['jepa'] - results['regular'],
            'hypothesis_supported': results['jepa'] > results['regular'],
            'max_examples': args.max_examples,
            'pca_components': args.pca_components
        }, f, indent=2)
    print("\nResults saved to: linearity_results.json")

if __name__ == "__main__":
    main()
