import json, glob
T = 'hybrid_adaptive/results/qwen/traces/arm1_trace.json'
h = json.load(open(T, encoding='utf-8'))['history']
f = sorted(glob.glob('hybrid_adaptive/results/qwen/checkpoints/arm1_hybrid_retrieval/**/trainer_state.json', recursive=True), key=lambda p: json.load(open(p))['global_step'])[-1]
s = [x for x in json.load(open(f))['log_history'] if 'loss' in x]
print('TRACE near the lambda cliff (step, lambda, lm_loss):')
for e in h:
    if 300 <= e['step'] <= 560: print('  ', e['step'], f"{e['lambda']:.3e}", round(e['lm_loss'], 4))
print('step gaps in trace:', sorted({b['step'] - a['step'] for a, b in zip(h, h[1:])}))
print('TRAINER log (step, loss): first 8, then 450-550')
for e in s[:8]: print('  ', e['step'], round(e['loss'], 4))
for e in s:
    if 450 <= e['step'] <= 550: print('  ', e['step'], round(e['loss'], 4))
