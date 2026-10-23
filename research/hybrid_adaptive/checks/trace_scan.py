import json
base = 'hybrid_adaptive/results/qwen/traces/'
for n in ['arm1_trace', 'spider_trace', 'gsm8k_trace', 'rotten_tomatoes_trace']:
    d = json.load(open(base + n + '.json', encoding='utf-8'))
    h = d['history']
    print('\n==', n, '| config:', d['config'])
    print('   entries:', len(h), '| steps:', h[0]['step'], '->', h[-1]['step'])
    print('   nonzero jepa_loss:', sum(1 for e in h if e['jepa_loss'] != 0), '| max =', max(e['jepa_loss'] for e in h))
    print('   nonzero S        :', sum(1 for e in h if e['S'] != 0), '| max =', max(e['S'] for e in h))
    print('   nonzero S_bar    :', sum(1 for e in h if e['S_bar'] != 0), '| max =', max(e['S_bar'] for e in h))
    half = next((e['step'] for e in h if e['lambda'] < 0.5), None)
    tiny = next((e['step'] for e in h if e['lambda'] < 1e-6), None)
    print('   first step with lambda<0.5:', half, '| lambda<1e-6:', tiny)
