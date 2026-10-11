import runpy, sys, time, torch
sys.argv = ['finetune_hybrid.py'] + sys.argv[1:]
t = time.time()
try:
    runpy.run_path('hybrid_adaptive/finetune_hybrid.py', run_name='__main__')
except torch.cuda.OutOfMemoryError:
    print('RESULT: OOM')
finally:
    print(f'PEAK_MB={torch.cuda.max_memory_allocated()/2**20:.0f} SECONDS={time.time()-t:.0f}')
