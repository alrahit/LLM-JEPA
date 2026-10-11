import json
R = 'hybrid_adaptive/results/'
def g(p): return [x['generated_text'].strip() for x in json.load(open(p, encoding='utf-8'))['details']]
pairs = {'SYNTH': (R+'final_synth_linalign_eval_results.json', R+'final_synth_sft_eval_results.json'),
         'SPIDER': (R+'final_spider_linalign_eval_results.json', R+'spider_sft_true_eval_results.json')}
for n, (a, b) in pairs.items():
    A, B = g(a), g(b)
    print(n, 'identical generated text:', sum(x == y for x, y in zip(A, B)), '/', len(A))
