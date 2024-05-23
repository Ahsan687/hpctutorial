import numpy as np
import multiprocessing as mp
from time import time
import sys

if(len(sys.argv)==3):
    N_Proc = int(sys.argv[1])
    N_evals = int(sys.argv[2])
else: 
    N_Proc = int(input("Specify number of cores (e.g., 4; use 1 for serial job): "))
    N_evals = int(input("Specify number of evaluations (e.g., 100): "))

Multiplier = 2


def do_this_job(n,mult):
    a = 0
    for i in range(100000*(n+1)):
        a += 1
    a *= mult
    return a,n

if N_Proc == 1:
    start_time = time()
    print('serial calculation')
    output = []
    for n in range(N_evals):
        do_this_job(n,Multiplier)
    print('time = {0:.2e} sec'.format(time() - start_time))
else:
    start_time = time()
    output = []
    pool = mp.Pool(processes=N_Proc)
    print('pool pooled')
    results = [pool.apply_async(do_this_job,args=(n,Multiplier)) for n in range(N_evals)]
    print('results setup')
    for n in range(N_evals):
        results[n].get()
    print('results got')
    pool.close()
    pool.join()
    print('pool closed')
    # print (output)
    print('time = {0:.2e} sec'.format(time() - start_time))
