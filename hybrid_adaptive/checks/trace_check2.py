import json
base = 'hybrid_adaptive/results/qwen/traces/'
for n in ['arm1_trace', 'spider_trace', 'gsm8k_trace', 'rotten_tomatoes_trace']:
    d = json.load(open(base + n + '.json', encoding='utf-8'))
    print('\n==', n, '| type:', type(d).__name__, '| top keys:', list(d.keys()) if isinstance(d, dict) else f'list len {len(d)}')
    h = d if isinstance(d, list) else None
    if h is None:
        for k, v in d.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                h = v; print('   history key:', k); break
            if not isinstance(v, (list, dict)): print('   ', k, '=', v)
    if not h:
        print('   (no list-of-dicts history found)'); continue
    print('   entries:', len(h), '| fields:', list(h[0].keys()))
    lam = [k for k in h[0] if 'lam' in k.lower() or 'lbd' in k.lower()]
    for k in lam:
        vals = [e[k] for e in h if isinstance(e.get(k), (int, float))]
        if vals:
            idx = [int(i * (len(vals) - 1) / 9) for i in range(10)]
            print(f'   {k}: first={vals[0]:.4f} min={min(vals):.4f} max={max(vals):.4f} last={vals[-1]:.4f}')
            print('      trajectory (10 pts):', [round(vals[i], 3) for i in idx])
    print('   first entry:', h[0]); print('   last entry :', h[-1])
