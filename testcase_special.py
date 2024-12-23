from plot_utils import *
from math_utils import *
from low_rank_approximation import *

methods = 'sketching_g'
epsilon = 1e-08
ks = [
    1,
    5,
    20,
    50
]
dims = [
    100,
    200
]
function = [create_random_diagonal_matrix, 
          create_random_orthogonal_matrix,
          create_random_symmetric_matrix,
          create_random_upper_triangular_matrix]
function_names = ['diagonal', 'orthogonal', 'symmetric', 'upper_triangular']


for k in ks:
    for dim in dims:
        for i, fun in enumerate(function):
            
            A = fun(dim)
            start(A, k, 'special', f'{dim}x{dim}_{function_names[i]}', f'{k}', methods, epsilon=epsilon)
            continue
        
for k in ks:
    for dim in dims:
        

        A = create_random_matrix_with_eigenvalues(dim, np.linspace(-1.0, 1.0, dim))
        start(A, k, 'special', f'{dim}x{dim}_c1', f'{k}', methods, epsilon=epsilon)
        
        A = create_random_matrix_with_eigenvalues(dim, np.linspace(-1.0, 10.0, dim))
        start(A, k, 'special', f'{dim}x{dim}_c10', f'{k}', methods, epsilon=epsilon)
        
        A = create_random_matrix_with_eigenvalues(dim, np.linspace(-1.0, 50.0, dim))
        start(A, k, 'special', f'{dim}x{dim}_c50', f'{k}', methods, epsilon=epsilon)
        
        A = create_random_matrix_with_eigenvalues(dim, np.linspace(-1.0, 100.0, dim))
        start(A, k, 'special', f'{dim}x{dim}_c100', f'{k}', methods, epsilon=epsilon)
        
        A = create_random_matrix_with_eigenvalues(dim, np.linspace(-1.0, 1000.0, dim))
        start(A, k, 'special', f'{dim}x{dim}_c1000', f'{k}', methods, epsilon=epsilon)
        
        
        eig = np.linspace(-1.0, 1, dim)
        eig[-1] = 50.0
        A = create_random_matrix_with_eigenvalues(dim, eig)
        start(A, k, 'special', f'{dim}x{dim}_c50outlier', f'{k}', methods, epsilon=epsilon)