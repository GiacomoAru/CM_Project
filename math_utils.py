import numpy as np
from scipy.linalg import solve, solve_triangular
from scipy.linalg.lapack import dgeqrf, dorgqr
from scipy.sparse.linalg import svds
import numpy as np
import pandas as pd
import json 
from pathlib import Path
import random

def thin_qr_factorization_OTS(A, threshold = None):
    """
    Perform a thin QR factorization using the off-the-shelf LAPACK routines.
    
    Parameters:
    A (ndarray): The m x n input matrix (m >= n).
    
    Returns:
    R (ndarray): The n x n upper triangular matrix.
    V (list): A list containing the Householder vectors.
    """
    m, n = A.shape

    if not threshold:
        eps = np.finfo(float).eps
        threshold = eps * np.max(np.abs(A))
    
    # Compute QR factorization using dgeqrf
    # Returns R (in the upper triangle) and the Householder vectors (in the lower triangle)
    # Step 1: Compute QR factorization using dgeqrf
    result = dgeqrf(A, overwrite_a=False)
    QR = result[0]
    
    # Extract R (upper triangular matrix)
    R = np.triu(QR[:n, :n])
    
    # Extract Householder vectors into a list
    householder_vectors = []
    for k in range(n):

        v = QR[k:, k]

        v[0] = 1.0  # Set the implicit 1 for the Householder vector

        # Normalize the vector
        if np.max(np.abs(v)) > threshold:  
            v /= np.linalg.norm(v)
        else:
            v = np.zeros_like(v)

        householder_vectors.append(v)
    
    return R, householder_vectors

def thin_qr_factorization_OTS_No_Householder(A):
    """
    Perform a thin QR factorization using the off-the-shelf numpy algorithm.
    
    Parameters:
    A (ndarray): The m x n input matrix (m >= n).
    
    Returns:
    Q (ndarray): the thin  m x n orthogonal matrix
    R (ndarray): The n x n upper triangular matrix.
    """
    Q, R = np.linalg.qr(A, mode='reduced')
    
    return Q, R 

def backward_substitution_OTS(A, T):
    #TODO check errors
    """
    Perform backward substitution to solve XT = A row by row using scipy.linalg.solve_triangular.

    Parameters:
        A (numpy.ndarray): Input matrix of size n x k.
        T (numpy.ndarray): Lower triangular matrix of size k x k.

    Returns:
        X (numpy.ndarray): Solution matrix of size n x k.
    """

    X = solve_triangular(T, A.T).T

    return X

def objective_function(A, U, V):
    """
    Computes the Frobenius norm of the matrix difference (A - U * V^t), where:
    - A is the matrix to compare,
    - U and V are matrices or vectors such that their multiplication is valid.

    Parameters:
        A (numpy.ndarray): The matrix of shape (m, n).
        U (numpy.ndarray): A matrix or vector with shape (m, k).
        V (numpy.ndarray): A matrix or vector with shape (k, n).

    Returns:
        float: The Frobenius norm of (A - U * V).
    """
    # Compute the matrix product U * V
    UV = np.dot(U, np.transpose(V))
    
    # Compute the difference A - U * V
    difference = A - UV
    
    # Compute the Frobenius norm of the difference
    frobenius_norm = np.linalg.norm(difference, 'fro')
    
    return frobenius_norm

def create_diagonal_matrix_from_eigenvalues(eigenvalues):
    """
    Crea una matrice diagonale con gli autovalori specificati.
    
    Args:
        eigenvalues (list or np.ndarray): Lista o array di autovalori desiderati.
    
    Returns:
        numpy.ndarray: Matrizzazione diagonale con gli autovalori specificati.
    """
    # Creare una matrice diagonale con gli autovalori
    diagonal_matrix = np.diag(eigenvalues)
    
    return diagonal_matrix

def create_symmetric_matrix_from_eigenvalues(eigenvalues):
    """
    Crea una matrice simmetrica con gli autovalori specificati, ma non diagonale.
    
    Args:
        eigenvalues (list or np.ndarray): Lista di autovalori desiderati.
    
    Returns:
        numpy.ndarray: Matrizzazione simmetrica con gli autovalori specificati, ma non diagonale.
    """
    # Numero di autovalori
    size = len(eigenvalues)
    
    # Creiamo una matrice ortogonale Q (random)
    Q, _ = np.linalg.qr(np.random.rand(size, size))  # Q è una matrice ortogonale
    
    # Creiamo la matrice diagonale con gli autovalori
    Lambda = np.diag(eigenvalues)
    
    # Costruire la matrice simmetrica A = Q * Lambda * Q.T
    A = Q @ Lambda @ Q.T
    
    return A

def upscale_bilinear(matrix, output_rows, output_cols):
    """
    Upscales a 2D matrix using bilinear interpolation to the specified output dimensions.
    
    Parameters:
        matrix (2D numpy array): Input matrix to be upscaled.
        output_rows (int): The number of rows in the upscaled matrix.
        output_cols (int): The number of columns in the upscaled matrix.
        
    Returns:
        2D numpy array: The upscaled matrix.
    """
    input_rows, input_cols = matrix.shape
    scale_factor_x = output_rows / input_rows
    scale_factor_y = output_cols / input_cols

    # Create an output matrix of the desired size
    upscaled_matrix = np.zeros((output_rows, output_cols), dtype=float)

    for i in range(output_rows):
        for j in range(output_cols):
            # Map the output pixel to the fractional input coordinates
            x = i / scale_factor_x
            y = j / scale_factor_y
            
            # Get the integer part and the fractional part
            x0 = int(np.floor(x))
            y0 = int(np.floor(y))
            x1 = min(x0 + 1, input_rows - 1)
            y1 = min(y0 + 1, input_cols - 1)

            # Compute the weights for interpolation
            wx1, wx0 = x - x0, 1 - (x - x0)
            wy1, wy0 = y - y0, 1 - (y - y0)

            # Perform bilinear interpolation
            upscaled_matrix[i, j] = (
                matrix[x0, y0] * wx0 * wy0 +
                matrix[x0, y1] * wx0 * wy1 +
                matrix[x1, y0] * wx1 * wy0 +
                matrix[x1, y1] * wx1 * wy1
            )

    return upscaled_matrix

def upscale_nearest_neighbor(matrix, output_rows, output_cols):
    """
    Upscales a 2D matrix using nearest neighbor interpolation to the specified output dimensions.
    
    Parameters:
        matrix (2D numpy array): Input matrix to be upscaled.
        output_rows (int): The number of rows in the upscaled matrix.
        output_cols (int): The number of columns in the upscaled matrix.
        
    Returns:
        2D numpy array: The upscaled matrix.
    """
    input_rows, input_cols = matrix.shape
    scale_factor_x = output_rows / input_rows
    scale_factor_y = output_cols / input_cols

    # Create an output matrix of the desired size
    upscaled_matrix = np.zeros((output_rows, output_cols), dtype=matrix.dtype)

    for i in range(output_rows):
        for j in range(output_cols):
            # Map the output pixel to the nearest input pixel
            nearest_row = int(i / scale_factor_x)
            nearest_col = int(j / scale_factor_y)
            upscaled_matrix[i, j] = matrix[nearest_row, nearest_col]

    return upscaled_matrix

def create_random_matrix_with_rank(seed, k, m, n):
    """
    Create a random matrix of rank exactly k using matrix multiplication.
    
    Parameters:
        seed (int): The seed for the random number generator.
        k (int): The desired rank of the matrix.
        m (int): The number of rows of the resulting matrix.
        n (int): The number of columns of the resulting matrix.
        
    Returns:
        numpy.ndarray: A random matrix of shape (m, n) with rank exactly k.
    """
    np.random.seed(seed)
    
    # Create two random matrices of appropriate sizes
    U = np.random.randn(m, k)
    V = np.random.randn(k, n)
    
    # Compute the product to get a matrix of rank k
    A = np.dot(U, V)
    
    return A



def create_random_diagonal_matrix(k, mean=0, std=1):
    """
    Create a k x k diagonal matrix with diagonal values drawn from a normal distribution.
    
    Parameters:
        k (int): The size of the matrix (k x k).
        mean (float): The mean of the normal distribution.
        std (float): The standard deviation of the normal distribution.
        
    Returns:
        numpy.ndarray: A k x k diagonal matrix with values drawn from a normal distribution.
    """
    # Draw k values from a normal distribution
    diagonal_values = np.random.normal(mean, std, k)
    
    # Create a diagonal matrix with the drawn values
    diagonal_matrix = np.diag(diagonal_values)
    
    return diagonal_matrix

def create_random_orthogonal_matrix(k, mean=0, std=1):
    """
    Create a random k x k orthogonal matrix using the QR decomposition of a random matrix.
    
    Parameters:
        k (int): The size of the orthogonal matrix (k x k).
        
    Returns:
        numpy.ndarray: A k x k orthogonal matrix.
    """
    # Create a random k x k matrix with values drawn from a normal distribution
    random_matrix = np.random.normal(mean, std, (k, k))
    
    # Perform QR decomposition to obtain an orthogonal matrix Q
    Q, _ = np.linalg.qr(random_matrix)
    
    return Q

def create_random_symmetric_matrix(k, mean=0, std=1):
    """
    Create a k x k symmetric matrix with values drawn from a Gaussian distribution.
    
    Parameters:
        k (int): The size of the matrix (k x k).
        mean (float): The mean of the Gaussian distribution.
        std (float): The standard deviation of the Gaussian distribution.
        
    Returns:
        numpy.ndarray: A k x k symmetric matrix with values drawn from a Gaussian distribution.
    """
    # Create a random k x k matrix with values drawn from a Gaussian distribution
    random_matrix = np.random.normal(mean, std, (k, k))
    
    # Make the matrix symmetric
    symmetric_matrix = (random_matrix + random_matrix.T) / 2
    
    return symmetric_matrix

def create_random_upper_triangular_matrix(k, mean=0, std=1):
    """
    Create a k x k upper triangular matrix with values drawn from a normal distribution.
    
    Parameters:
        k (int): The size of the matrix (k x k).
        mean (float): The mean of the normal distribution.
        std (float): The standard deviation of the normal distribution.
        
    Returns:
        numpy.ndarray: A k x k upper triangular matrix with values drawn from a normal distribution.
    """
    # Create a random k x k matrix with values drawn from a normal distribution
    random_matrix = np.random.normal(mean, std, (k, k))
    
    # Make the matrix upper triangular
    upper_triangular_matrix = np.triu(random_matrix)
    
    return upper_triangular_matrix

def create_random_matrix_with_eigenvalues(k, eigenvalues, mean=0, std=1):
    """
    Create a k x k random matrix with eigenvalues equal to the ones given in input.
    
    Parameters:
        k (int): The size of the matrix (k x k).
        eigenvalues (list or np.ndarray): The desired eigenvalues.
        mean (float): The mean of the normal distribution for generating the random matrix.
        std (float): The standard deviation of the normal distribution for generating the random matrix.
        
    Returns:
        numpy.ndarray: A k x k random matrix with the specified eigenvalues.
    """
    # Create a random k x k matrix with values drawn from a normal distribution
    random_matrix = np.random.normal(mean, std, (k, k))
    
    # Perform QR decomposition to obtain an orthogonal matrix Q
    Q, _ = np.linalg.qr(random_matrix)
    
    # Create a diagonal matrix with the given eigenvalues
    Lambda = np.diag(eigenvalues)
    
    # Construct the random matrix with the specified eigenvalues
    random_matrix_with_eigenvalues = Q @ Lambda @ Q.T
    
    return random_matrix_with_eigenvalues




def keep_n_files(folder, n):
    # Convert folder to a Path object
    folder_path = Path(folder)
    
    # Get a list of all files in the folder (excluding directories)
    files = [f for f in folder_path.iterdir() if f.is_file()]

    # If the folder contains less than or equal to n files, no need to delete anything
    if len(files) <= n:
        print(f"The folder contains {len(files)} files, which is less than or equal to {n}. No files need to be removed.")
        return

    # Select n random files to keep
    files_to_keep = random.sample(files, n)

    # Remove all files except the ones selected
    for file in files:
        if file not in files_to_keep:
            file.unlink()  # Delete the file
            print(f"File removed: {file.name}")

    print(f"{n} random files have been kept. The others have been deleted.")
    
    
    
def load_matrices(c_name, m_name, t_name, matrices={'U', 'V', 'A', 'V_0'}, test_dir='./data/test'):
    ret = {}
    for el in matrices:
        ret[el] = np.load(f'{test_dir}/{c_name}/{m_name}/{t_name}/{el}.npy')
    return ret

def load_global_df(dataframe_path = './data/global_data.csv', filter={}):
    df = pd.read_csv(dataframe_path)
   
    for fun in filter:
        df = df[filter[fun](df[fun])] 
        
    return df

def compute_global_stats_df(test_dir = "./data/test", save_path = './data/global_data.csv'):
    main_folder = Path(test_dir)
    global_df = {
        'c_name':[], 'm_name':[], 't_name':[], 
        'init_method':[], 'epsilon':[],
        'pc_name':[], 'date':[],
        'm':[], 'n':[], 'k':[], 
        'iteration':[], 'exec_time':[], 
        'qr_time':[], 'manip_time':[], 'bw_time':[],
        'obj_fun':[], 'obj_fun_rel':[],
        'U_norm':[], 'V_norm':[]
    }
    
    # Iterate over all directories and subdirectories
    for subfolder in main_folder.rglob('*'):
        # Check if the path is a directory and is at the third level
        if subfolder.is_dir() and len(subfolder.relative_to(main_folder).parts) == 3:
            dummy_df = pd.read_csv((subfolder / 'data.csv').absolute())
            with open(subfolder / 'input_values.json', 'r') as f:
                input_values = json.loads(f.read())
            with open(subfolder / 'pc_info.json', 'r') as f:
                pc_info = json.loads(f.read())
                
            global_df['c_name'].append(input_values['c_name'])
            global_df['m_name'].append(input_values['m_name'])
            global_df['t_name'].append(input_values['t_name'])

            global_df['init_method'].append(input_values['init_method'])
            global_df['epsilon'].append(input_values['epsilon'])
            
            global_df['pc_name'].append(pc_info['pc_name'])
            global_df['date'].append(input_values['date'])
            
            global_df['iteration'].append(dummy_df['iteration_id'].values[-1])
            
            global_df['qr_time'].append(np.mean(dummy_df['qr_time'].values))
            global_df['manip_time'].append(np.mean(dummy_df['manip_time'].values))
            global_df['bw_time'].append(np.mean(dummy_df['bw_time'].values))
            global_df['exec_time'].append(global_df['qr_time'][-1] + global_df['manip_time'][-1] + global_df['bw_time'][-1])
    
            global_df['obj_fun'].append(dummy_df['obj_fun'].values[-1])
            # global_df['UV_norm'].append(dummy_df['UV_norm'].values[-1])
            global_df['U_norm'].append(dummy_df['U_norm'].values[-1])
            global_df['V_norm'].append(dummy_df['V_norm'].values[-1])
            
            A = np.load((subfolder / 'A.npy').absolute())
            U = np.load((subfolder / 'U.npy').absolute())

            global_df['m'].append(A.shape[0])
            global_df['n'].append(A.shape[1])
            global_df['k'].append(U.shape[1])
            
            global_df['obj_fun_rel'].append(global_df['obj_fun'][-1] / np.linalg.norm(A, 'fro'))
    
    global_df = pd.DataFrame(global_df)       
    global_df.to_csv(save_path, index=False)


def compute_global_minimum(A, k):

    U, S, VT = np.linalg.svd(A, full_matrices=False)

    # Ricostruzione della matrice a rango ridotto
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    VT_k = VT[:k, :]
    A_k = U_k @ S_k @ VT_k

    return A_k

def compute_global_minimum_sparse(A, k):

    # Decomposizione SVD troncata
    U, S, VT = svds(A, k=k)
    A_k = U @ np.diag(S) @ VT
    return A_k

def compare_solution_with_global_min(A, k, UVT):

    A_k = compute_global_minimum(A, k)
    error_relative = np.linalg.norm(A - UVT, 'fro') / np.linalg.norm(A - A_k, 'fro')