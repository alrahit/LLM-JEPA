import sys, json
sys.path.insert(0, 'hybrid_adaptive/scripts')
import parse_structural as p
from math import comb

def load(path):
    d = json.load(open(path, encoding='utf-8'))['details']
    ok, prompts = [], []
    for it in d:
        g = p.extract_structural_content(it.get('ground_truth', ''), True)
        q = p.extract_structural_content(it.get('generated_text', ''), False)
        ok.append(bool(g and q and (g == q or g in q)))
        prompts.append(it.get('prompt', ''))
    return ok, prompts

a, pa = load(sys.argv[1]); b, pb = load(sys.argv[2])
print("Same test order:", pa == pb, "| N =", len(a))
print(f"A (LinAlign) acc over ALL samples: {sum(a)/len(a)*100:.2f}% ({sum(a)})")
print(f"B (SFT)      acc over ALL samples: {sum(b)/len(b)*100:.2f}% ({sum(b)})")
a_only = sum(1 for x, y in zip(a, b) if x and not y)
b_only = sum(1 for x, y in zip(a, b) if y and not x)
n = a_only + b_only; k = min(a_only, b_only)
pval = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
print(f"Only A correct: {a_only} | Only B correct: {b_only} | McNemar exact p = {pval:.4f}")
