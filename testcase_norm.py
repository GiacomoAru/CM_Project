import numpy as np
from plot_utils import *
from math_utils import *
from low_rank_approximation import *


epsilon = 1e-15
ks = [
    1,
    5,
    10,
    20
]

norm_mul = [10**x for x in range(16)]
for norm in norm_mul:
    for k in ks:
        
        A = np.random.normal(0,1,(100,100))

        V = np.transpose(np.random.normal(0, 1, (k, 100)) @ A)
        frob_A = np.linalg.norm(A, 'fro')
        frob_V = np.linalg.norm(V, 'fro')
        V_0 = (V/frob_V)*np.sqrt(frob_A*np.sqrt(k))*norm
    
        start(A, k, 'g_norm', f'100x100_normal', f'{k}_Vmul{norm}', epsilon=epsilon, V_0 = V_0)
