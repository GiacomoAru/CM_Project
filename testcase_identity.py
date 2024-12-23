import numpy as np
from plot_utils import *
from math_utils import *
from low_rank_approximation import *

method = 'rand_n'
epsilon = 1e-15
ks = [
    1,
    5,
    10,
    20
]


A = np.eye(100)
noises = [np.zeros_like(A)] + [np.random.normal(0, 10**-x, A.shape) for x in range(11)]
names = ['_0']  + [f'_{10**-x}' for x in range(11)]

for iter in range(2):
    for name, noise in zip(names, noises):
        for k in ks:
            start(A + noise, k, 'g_identity', f'100x100{name}',  
                    f'{iter}_{k}', method, epsilon=epsilon)  
            continue
