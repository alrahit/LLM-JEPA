import sys, json, random
from math import comb
sys.path.insert(0, 'hybrid_adaptive/scripts')
import parse_structural as p

R = 'hybrid_adaptive/results/'
CFG = {
  'SYNTH':  dict(train='datasets/synth_train.jsonl',  test='datasets/synth_test.jsonl',
                 ZS=R+'final_synth_zeroshot_eval_results.json',
                 SFT=R+'final_synth_sft_eval_results.json',
                 LA=R+'final_synth_linalign_eval_results.json'),
  'SPIDER': dict(train='datasets/spider_train.jsonl', test='datasets/spider_test.jsonl',
                 ZS=R+'final_spider_zeroshot_eval_results.json',
                 SFT=R+'spider_sft_true_eval_results.json',
                 LA=R+'final_spider_linalign_eval_results.json'),
}

def rows(path): return [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
def key(r): return json.dumps(r['messages'][:-1], ensure_ascii=False)

def load(path):
    d = json.load(open(path, encoding='utf-8'))['details']
    parsed, raw, prompts = [], [], []
    for it in d:
        g = p.extract_structural_content(it.get('ground_truth', ''), True)
        q = p.extract_structural_content(it.get('generated_text', ''), False)
        parsed.append(bool(g and q and (g == q or g in q)))
        raw.append(bool(it.get('is_correct')))
        prompts.append(it.get('prompt', ''))
    return parsed, raw, prompts

def mcnemar(a, b):
    ao = sum(x and not y for x, y in zip(a, b)); bo = sum(y and not x for x, y in zip(a, b))
    n = ao + bo; k = min(ao, bo)
    pv = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return ao, bo, pv

def ci(a, b, B=5000, seed=0):
    rng = random.Random(seed); n = len(a)
    d = [int(x) - int(y) for x, y in zip(a, b)]
    res = sorted(sum(rng.choices(d, k=n)) / n * 100 for _ in range(B))
    return res[int(0.025 * B)], res[int(0.975 * B)]

def cmp(na, a, nb, b):
    n = len(a); ao, bo, pv = mcnemar(a, b); lo, hi = ci(a, b)
    print(f"   {na:>3} - {nb:<3}: {(sum(a)-sum(b))/n*100:+6.2f} pts  95% CI [{lo:+.2f}, {hi:+.2f}]  only-{na}={ao} only-{nb}={bo}  McNemar p={pv:.4f}")

for name, c in CFG.items():
    te = rows(c['test']); trk = {key(r) for r in rows(c['train'])}
    tp = [r['messages'][-2]['content'] for r in te]
    leak = [key(r) in trk for r in te]
    arms = {}
    print(f"\n===== {name}  N={len(te)}  | test prompts also in train: {sum(leak)}")
    for a in ('ZS', 'SFT', 'LA'):
        pa, rw, pr = load(c[a])
        assert len(pa) == len(te), f"{a}: length mismatch"
        mism = sum(1 for x, y in zip(pr, tp) if x != y)
        arms[a] = pa
        print(f"   {a:>3}: raw EM {sum(rw)/len(rw)*100:6.2f}% ({sum(rw)}) | parsed, all N {sum(pa)/len(pa)*100:6.2f}% ({sum(pa)}) | order mismatches={mism} (must be 0)")
    print("  -- ALL samples (paired)")
    cmp('LA', arms['LA'], 'SFT', arms['SFT']); cmp('LA', arms['LA'], 'ZS', arms['ZS']); cmp('SFT', arms['SFT'], 'ZS', arms['ZS'])
    keep = [not l for l in leak]
    cl = {a: [x for x, k in zip(v, keep) if k] for a, v in arms.items()}
    print(f"  -- CLEAN samples (train-overlap removed, N={sum(keep)})")
    cmp('LA', cl['LA'], 'SFT', cl['SFT']); cmp('LA', cl['LA'], 'ZS', cl['ZS']); cmp('SFT', cl['SFT'], 'ZS', cl['ZS'])
