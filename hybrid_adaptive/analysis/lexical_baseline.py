r"""Lexical control for the retrieval probe: how well can plain word / character overlap
match a user message to its assistant answer, using the SAME protocol (random batches of B,
top-1 accuracy, 3 repetitions) as retrieval_probe2.py?

If this reaches the accuracy of the mid-layer embeddings, the pairing signal in those
embeddings may just be lexical overlap.

Run from the repo root (D:/LLM_JEPA):
  python hybrid_adaptive\analysis\lexical_baseline.py --dataset synth
"""
import argparse, json, re
import numpy as np


def read_pairs(path, n):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                m = json.loads(line)["messages"]
                out.append((m[1]["content"], m[2]["content"]))
            if len(out) >= n:
                break
    return out


def word_tokens(s):
    return re.findall(r"\w+|[^\w\s]", s.lower())


def char_ngrams(s, n=3):
    s = " " + re.sub(r"\s+", " ", s.lower()) + " "
    return [s[i:i + n] for i in range(max(1, len(s) - n + 1))]


def tfidf_pair(U_docs, A_docs):
    docs = U_docs + A_docs
    vocab = {}
    for d in docs:
        for t in set(d):
            vocab.setdefault(t, len(vocab))
    N = len(docs)
    X = np.zeros((N, len(vocab)), dtype=np.float32)
    df = np.zeros(len(vocab), dtype=np.float32)
    for i, d in enumerate(docs):
        for t in d:
            X[i, vocab[t]] += 1.0
        for t in set(d):
            df[vocab[t]] += 1.0
    idf = np.log((1.0 + N) / (1.0 + df)) + 1.0
    X = np.log1p(X) * idf
    half = len(U_docs)
    return X[:half], X[half:], len(vocab)


def prep(U, A, centered):
    if centered:
        U = U - U.mean(0, keepdims=True)
        A = A - A.mean(0, keepdims=True)
    U = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-8)
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-8)
    return U, A


def acc_at(U, A, B, reps=3, seed=0):
    N = U.shape[0]
    rng = np.random.default_rng(seed)
    hits = []
    for _ in range(reps):
        perm = rng.permutation(N)
        for s in range(0, N - B + 1, B):
            idx = perm[s:s + B]
            sim = U[idx] @ A[idx].T
            hits.append(float((sim.argmax(1) == np.arange(B)).mean()))
    return sum(hits) / len(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["synth", "spider", "gsm8k", "rotten_tomatoes"])
    ap.add_argument("--n", type=int, default=512)
    args = ap.parse_args()
    pairs = read_pairs(f"datasets/{args.dataset}_test.jsonl", args.n)
    print(f"{args.dataset}: {len(pairs)} test pairs. chance: acc16=0.062 acc64=0.016")
    print("example user   :", pairs[0][0][:120].replace("\n", " "))
    print("example answer :", pairs[0][1][:120].replace("\n", " "))
    for name, fn in (("word tf-idf", word_tokens), ("char-3gram tf-idf", char_ngrams)):
        Ud, Ad = [fn(u) for u, _ in pairs], [fn(a) for _, a in pairs]
        U, A, v = tfidf_pair(Ud, Ad)
        for centered in (False, True):
            Un, An = prep(U.copy(), A.copy(), centered)
            print(f"  {name:18s} vocab={v:6d} centered={str(centered):5s} acc16={acc_at(Un, An, 16):.3f} acc64={acc_at(Un, An, 64):.3f}")


if __name__ == "__main__":
    main()
