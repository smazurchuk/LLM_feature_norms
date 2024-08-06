#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:18:03 2024

This script runs all the LLM comparisons in parallel

@author: smazurchuk
"""
import subprocess
import multiprocessing as mp
import time

def run_script(idx_q,gpu):
    while idx_q.qsize() > 0:
        idx = idx_q.get()
        # Run command
        print(f'Idx: {idx}, g: {gpu}')
        command = f'python -u compute_LLM_comparisons.py {idx} {gpu} >> output_logs/{idx:03.0f}.txt 2>&1'
        subprocess.run(command, check=False, shell=True)
    return 

def main():
    start = time.time()
        
    # Create a pool of worker processes
    idx_q = mp.Queue()
    for i in range(56):
        idx_q.put(i)
    gpus = [f'{2*i},{(2*i)+1}' for i in range(8)]
    
    # Run the scripts in parallel
    procs = []
    for gpu in gpus:
        p = mp.Process(target=run_script, args=(idx_q,gpu))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print('Script Done!')
    print(f'Total time: {(time.time() - start)/60:0.2f} minutes')

if __name__ == "__main__":
    main()
