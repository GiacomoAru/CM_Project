import numpy as np
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import svds
import numpy as np
import pandas as pd
import json 
from pathlib import Path
import random

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
    """
    Keep a specified number of random files in a folder and delete the rest.
    Args:
        folder (str or Path): The path to the folder containing the files.
        n (int): The number of files to keep in the folder.
    Returns:
        None
    Notes:
        - If the folder contains less than or equal to n files, no files will be deleted.
        - The function randomly selects n files to keep and deletes the rest.
        - The function prints the names of the files that are removed and a message indicating how many files were kept.
    """
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
    """
    Load specified matrices from a given directory structure.

    Args:
        c_name (str): The name of the category or class directory.
        m_name (str): The name of the matrix directory.
        t_name (str): The name of the test directory.
        matrices (set, optional): A set of matrix names to load. Defaults to {'U', 'V', 'A', 'V_0'}.
        test_dir (str, optional): The base directory where the test data is stored. Defaults to './data/test'.

    Returns:
        dict: A dictionary where keys are matrix names and values are the loaded numpy arrays.
    """
    ret = {}
    for el in matrices:
        ret[el] = np.load(f'{test_dir}/{c_name}/{m_name}/{t_name}/{el}.npy')
    return ret

def load_global_df(dataframe_path = './data/global_data.csv', filter={},  new_col={}):
    """
    Load a global DataFrame from a CSV file, apply filters, and add new columns.
    Parameters:
    dataframe_path (str): The path to the CSV file to load. Default is './data/global_data.csv'.
    filter (dict): A dictionary where keys are column names and values are functions to filter the DataFrame.
    new_col (dict): A dictionary where keys are new column names and values are functions to generate the new columns.
    Returns:
    pandas.DataFrame: The processed DataFrame with applied filters and new columns.
    """
        
    df = pd.read_csv(dataframe_path)
   
    # Apply filters
    for fun in filter:
        try:
            df = df[filter[fun](df[fun])]
        except:
            continue
    for c in new_col:
        df[c] = new_col[c](df)
    for fun in filter:
        try:
            df = df[filter[fun](df[fun])]
        except:
            continue
        
    return df

def compute_global_stats_df(test_dir = "./data/test", save_path = './data/global_data.csv'):
    """
    Computes global statistics from test data and saves the results to a CSV file.
    Parameters:
    test_dir (str): The directory containing the test data. Default is "./data/test".
    save_path (str): The path where the resulting CSV file will be saved. Default is './data/global_data.csv'.
    The function iterates over all subdirectories within the specified test directory, 
    reads relevant data from CSV and JSON files, and computes various statistics. 
    These statistics include:
        - c_name, m_name, t_name: Names related to the test configuration.
        - init_method: Initialization method used.
        - epsilon: Epsilon value used in the test.
        - pc_name: Name of the PC used for the test.
        - date: Date of the test.
        - m, n, m*n: Dimensions of matrix A.
        - k: Inner dimention of U and V.
        - n_iteration: Number of iterations.
        - step_time, tot_time, qr_time, manip_time, bw_time: Timing information.
        - obj_fun_abs, obj_fun_rel: Objective function values.
        - error_ratio: Error ratio compared to global minimum (-1).
        - U_norm, V_norm, U_V_norm_diff: Norms of matrices U and V.
        - A_rank: Rank of matrix A.
    The computed statistics are saved to a CSV file at the specified save_path.
    """
    
    main_folder = Path(test_dir)
    global_df = {
        'c_name':[], 'm_name':[], 't_name':[], 
        'init_method':[], 'epsilon':[],
        'pc_name':[], 'date':[],
        'm':[], 'n':[], 'm*n':[], 'k':[], 
        'n_iteration':[], 'step_time':[], 'tot_time':[],
        'qr_time':[], 'manip_time':[], 'bw_time':[],
        'obj_fun_abs':[], 'obj_fun_rel':[], 'error_ratio':[],
        'U_norm':[], 'V_norm':[], 'U_V_norm_diff':[], 'A_rank':[],
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
                
            print(subfolder.absolute())
            
            A = np.load((subfolder / 'A.npy').absolute())
            U = np.load((subfolder / 'U.npy').absolute())
            V = np.load((subfolder / 'V.npy').absolute())
            
            global_df['c_name'].append(input_values['c_name'])
            global_df['m_name'].append(input_values['m_name'])
            global_df['t_name'].append(input_values['t_name'])

            global_df['init_method'].append(input_values['init_method'])
            global_df['epsilon'].append(input_values['epsilon'])
            
            global_df['pc_name'].append(pc_info['pc_name'])
            global_df['date'].append(input_values['date'])
            
            global_df['n_iteration'].append(dummy_df['iteration_id'].values[-1])
            
            global_df['qr_time'].append(np.mean(dummy_df['qr_time'].values))
            global_df['manip_time'].append(np.mean(dummy_df['manip_time'].values))
            global_df['bw_time'].append(np.mean(dummy_df['bw_time'].values))
            global_df['step_time'].append(global_df['qr_time'][-1] + global_df['manip_time'][-1] + global_df['bw_time'][-1])
            global_df['tot_time'].append(global_df['step_time'][-1]*global_df['n_iteration'][-1])
            
            global_df['obj_fun_abs'].append(dummy_df['obj_fun'].values[-1])
            global_df['obj_fun_rel'].append(global_df['obj_fun_abs'][-1] / np.linalg.norm(A, 'fro'))
            global_df['error_ratio'].append(compare_solution_with_global_min(A, U.shape[1], U @ V.T))
            
            global_df['U_norm'].append(dummy_df['U_norm'].values[-1])
            global_df['V_norm'].append(dummy_df['V_norm'].values[-1])
            global_df['U_V_norm_diff'].append(abs(global_df['U_norm'][-1] - global_df['V_norm'][-1]))
            global_df['A_rank'].append(np.linalg.matrix_rank(A))
            
            global_df['m'].append(A.shape[0])
            global_df['n'].append(A.shape[1])
            global_df['m*n'].append(global_df['m'][-1] * global_df['n'][-1])
            global_df['k'].append(U.shape[1])
            
            
    
    global_df = pd.DataFrame(global_df)       
    global_df.to_csv(save_path, index=False)



def compute_global_minimum(A, k):
    """
    Compute the rank-k approximation of matrix A using Singular Value Decomposition (SVD).
    Parameters:
        A (numpy.ndarray): The input matrix to be approximated.
        k (int): The rank for the approximation.
    Returns:
        numpy.ndarray: The rank-k approximation of the input matrix A.
    """

    U, S, VT = np.linalg.svd(A, full_matrices=False)

    # Ricostruzione della matrice a rango ridotto
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    VT_k = VT[:k, :]
    A_k = U_k @ S_k @ VT_k

    return A_k

def compare_solution_with_global_min(A, k, UVT):
    """
    Compares the given solution with the global minimum of matrix A.
    Parameters:
        A (numpy.ndarray): The original matrix.
        k (int): The rank for the global minimum computation.
        UVT (numpy.ndarray): The approximated solution matrix.
    Returns:
        float: The relative error between the given solution and the global minimum minus 1.
    """

    A_k = compute_global_minimum(A, k)
    error_relative = np.linalg.norm(A - UVT, 'fro') / np.linalg.norm(A - A_k, 'fro')

    return error_relative - 1