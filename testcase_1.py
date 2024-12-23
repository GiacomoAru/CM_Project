import numpy as np
from plot_utils import *
from math_utils import *
from low_rank_approximation import *

seeds = [123, 456]

methods = 'rand_n'
epsilon = 1e-15

ks = [
    1,
    5,
    10,
    20
]


A = np.eye(100)
noises = [np.zeros_like(A)] + [np.random.normal(0, 10**-x, A.shape) for x in range(9)].reverse()

for seed in seeds:
    for i, noise in enumerate(noises):
        for k in ks:
            for meth in methods:
                start(A + noise, k, 'identity', f'eye100x100_{10**-i}',  
                      f'{k}_{meth}_{seed}', meth, epsilon=epsilon, seed=seed)  
                continue