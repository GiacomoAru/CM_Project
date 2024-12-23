import numpy as np
from scipy.linalg import solve, solve_triangular
from scipy.linalg.lapack import dgeqrf, dorgqr
import numpy as np
import pandas as pd
import json 

def thin_qr_factorization_OTS(A):
    """
    Perform a thin QR factorization using the off-the-shelf LAPACK routines.
    
    Parameters:
    A (ndarray): The m x n input matrix (m >= n).
    
    Returns:
    R (ndarray): The n x n upper triangular matrix.
    V (list): A list containing the Householder vectors.
    """
    m, n = A.shape
    
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
        v /= np.linalg.norm(v)

        householder_vectors.append(v)
    
    return R, householder_vectors

def thin_qr_factorization_OTS_2(A):
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

def upscale_bilinear(matrix, scale_factor):
    """
    Upscales a 2D matrix using bilinear interpolation.
    
    Parameters:
        matrix (2D numpy array): Input matrix to be upscaled.
        scale_factor (int): The factor by which to upscale the matrix.
        
    Returns:
        2D numpy array: The upscaled matrix.
    """
    if scale_factor <= 0:
        raise ValueError("Scale factor must be a positive integer.")
    
    input_rows, input_cols = matrix.shape
    output_rows, output_cols = input_rows * scale_factor, input_cols * scale_factor

    # Create an output matrix of the desired size
    upscaled_matrix = np.zeros((output_rows, output_cols), dtype=float)

    for i in range(output_rows):
        for j in range(output_cols):
            # Map the output pixel to the fractional input coordinates
            x = i / scale_factor
            y = j / scale_factor
            
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

def upscale_nearest_neighbor(matrix, scale_factor):
    """
    Upscales a 2D matrix using nearest neighbor interpolation.
    
    Parameters:
        matrix (2D numpy array): Input matrix to be upscaled.
        scale_factor (int): The factor by which to upscale the matrix.
        
    Returns:
        2D numpy array: The upscaled matrix.
    """
    if scale_factor <= 0:
        raise ValueError("Scale factor must be a positive integer.")
    
    input_rows, input_cols = matrix.shape
    output_rows, output_cols = input_rows * scale_factor, input_cols * scale_factor

    # Create an output matrix of the desired size
    upscaled_matrix = np.zeros((output_rows, output_cols), dtype=matrix.dtype)

    for i in range(output_rows):
        for j in range(output_cols):
            # Map the output pixel to the nearest input pixel
            nearest_row = i // scale_factor
            nearest_col = j // scale_factor
            upscaled_matrix[i, j] = matrix[nearest_row, nearest_col]

    return upscaled_matrix
   
    
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
    
    
    
def load_matrices(c_name, m_name, t_name, matrices={'U', 'V', 'A', 'V_0'}):
    ret = {}
    for el in matrices:
        ret[el] = np.load(f'./data/test/{c_name}/{m_name}/{t_name}/{el}.npy')
    return ret

def load_global_df(filter={}):
    df = pd.read_csv('./data/global_data.csv')
   
    for fun in filter:
        df = df[filter[fun](df[fun])] 
        
    return df

def compute_global_stats_df():
    main_folder = Path("./data/test")
    global_df = {
        'c_name':[], 'm_name':[], 't_name':[], 
        'init_method':[], 'epsilon':[],
        'pc_name':[], 'date':[],
        'm':[], 'n':[], 'k':[], 
        'iteration':[], 'exec_time':[], 
        'qr_time':[], 'manip_time':[], 'bw_time':[],
        'obj_fun':[], # 'UV_norm':[],
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
    
    global_df = pd.DataFrame(global_df)       
    global_df.to_csv('./data/global_data.csv', index=False)

