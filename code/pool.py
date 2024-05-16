import numpy as np
import multiprocessing as mp
from time import time

N_Proc = 4
N_evals = 20
Multiplier = 2

# will modify soon to allow command-line arguments

output = []

def do_this_job(n,mult):
    a = 0
    for i in range(1000000*(n+1)):
        a += 1
    a *= mult
    return a,n

start_time = time()
pool = mp.Pool(processes=N_Proc)
#
print('pool pooled')
results = [pool.apply_async(do_this_job,args=(n,Multiplier)) for n in range(N_evals)]
print('results setup')
#
for n in range(N_evals):
    output.append(results[n].get())
#
print('results got')
pool.close()
pool.join()
print('pool closed')

print('time = {0:.2e} sec'.format(time() - start_time))

print (output)
