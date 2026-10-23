"""compare.py - paired comparison of two eval json files made on the same examples.
usage: python v2\\compare.py A_eval.json B_eval.json      (difference reported as B - A, metric = strict execution match)
"""
import json
import math
import random
import sys


def load(p):
    return json.load(open(p, encoding="utf-8"))["examples"]


a, b = load(sys.argv[1]), load(sys.argv[2])
assert len(a) == len(b), "different number of examples"
assert all(x["gold"] == y["gold"] and x["db_id"] == y["db_id"] for x, y in zip(a, b)), "examples differ"
xa = [int(e["strict"]) for e in a]
xb = [int(e["strict"]) for e in b]
n = len(a)
only_a = sum(1 for p, q in zip(xa, xb) if p == 1 and q == 0)
only_b = sum(1 for p, q in zip(xa, xb) if p == 0 and q == 1)
m = only_a + only_b
if m == 0:
    pval = 1.0
else:
    k = min(only_a, only_b)
    pval = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)  # exact two-sided McNemar
d = [q - p for p, q in zip(xa, xb)]
rng = random.Random(0)
boots = sorted(sum(rng.choices(d, k=n)) / n for _ in range(10000))
lo, hi = boots[249], boots[9749]
print(f"n={n}")
print(f"A: {100 * sum(xa) / n:.2f}%   B: {100 * sum(xb) / n:.2f}%   B-A: {100 * (sum(xb) - sum(xa)) / n:+.2f} pts")
print(f"paired bootstrap 95% CI of B-A: [{100 * lo:+.2f}, {100 * hi:+.2f}] pts")
print(f"only A correct: {only_a}   only B correct: {only_b}   exact McNemar p (two-sided): {pval:.4f}")
