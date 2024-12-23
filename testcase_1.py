import numpy as np
from plot_utils import *
from math_utils import *
from low_rank_approximation import *

seed = 123
seeds = [123, 456, 789, 101112]

methods = [
    'rand_n',           # Normal Distribution
    'sketching_g',      # Sketching with Gaussian Distribution
    'sketching_b',      # Sketching with Bernoulli Distribution
]
epsilon = 1e-08

ks = [
    1,
    5,
    10,
    20
]


noises = []
A = add_gaussian_noise(np.eye(100), 0, 1e-07)
for seed in seeds:
    for k in ks:
        for meth in methods:
            start(A, k, 'diagonal', 'eighen_bad_bad_250x250',  f'{k}_{meth}_{epsilon}', meth, epsilon=epsilon, seed=seed)