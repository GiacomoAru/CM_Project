from plot_utils import *
from math_utils import *
from low_rank_approximation import *

methods = 'sketching_b'
epsilon = 1e-15
ks = [
    1,
    5,
    20,
    50
]
ranks = [
    1,
    2,
    5,
    10,
    20,
    50,
    75
]

for iter in range(2):
    for k in ks:
        for rank in ranks:
            A = create_random_matrix_with_rank(None, rank, 100, 100)
            start(A, k, 'low_rank', f'100x100_{rank}', f'{iter}_{k}', methods, epsilon=epsilon)
            continue