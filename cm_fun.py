import numpy as np
import time
import pandas as pd
import os

#TODO: corretta
def thin_qr_factorization(A):
    """
    Perform the Thin QR factorization using Householder reflections.

    Parameters:
        A (numpy.ndarray): Input matrix of size m x n (m >= n).

    Returns:
        R (numpy.ndarray): Square upper triangular matrix of size n x n.
        householder_vectors (list): List of n Householder vectors.
    """
    m, n = A.shape
    A = A.copy()  # To avoid modifying the original matrix
    householder_vectors = []

    for k in range(n):
        # Extract the column vector to be reflected
        x = A[k:m, k]

        # Compute the Householder vector vk
        e1 = np.zeros_like(x)
        e1[0] = np.linalg.norm(x) if x[0] >= 0 else -np.linalg.norm(x)
        vk = x + e1
        vk /= np.linalg.norm(vk)

        # Store the Householder vector
        householder_vectors.append(vk)

        # Update the submatrix A[k:m, k:n]
        A[k:m, k:n] -= 2 * np.outer(vk, vk @ A[k:m, k:n])

    # Extract the upper triangular part as R (size n x n)
    R = np.triu(A[:n, :n])
    
    return R, householder_vectors

#TODO: corretta
def backward_substitution(A, T):
    """
    Perform backward substitution to solve XT = A row by row.

    Parameters:
        A (numpy.ndarray): Input matrix of size n x k.
        T (numpy.ndarray): Lower triangular matrix of size k x k.

    Returns:
        X (numpy.ndarray): Solution matrix of size n x k.
    """
    n, k = A.shape
    X = np.zeros_like(A)

    # for each column of X
    for i in range(k-1, -1, -1):
        # Solve for each row of X simultaneously
        X[:,i] = (A[:,i] - np.sum( (T[i+1:,i] * X[:,i+1:]), 1)) * (1/T[i,i])

    return X

#TODO: Funzione corretta
def apply_householder_transformations(A, householder_vectors):
    """
    Applies a series of Householder transformations to the matrix A using the given list of Householder vectors.

    Parameters:
        A (numpy.ndarray): The input matrix of shape (m, n).
        householder_vectors (list of numpy.ndarray): A list of Householder vectors. 
            The i-th vector corresponds to the i-th Householder transformation.

    Returns:
        numpy.ndarray: The matrix A after applying the Householder transformations.
    """
    # Copy A to avoid modifying the original matrix
    A = A.copy()
    
    for i, u in enumerate(householder_vectors):

        # Extract the submatrix of interest
        submatrix = A[:, i:]
        
        # Update the submatrix using the Householder transformation formula
        submatrix -= 2.0 * np.outer(np.dot(submatrix, u), u)

        # Update A with the transformed submatrix
        A[:, i:] = submatrix

    return A[:,:len(householder_vectors)]


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

def get_starting_matrix(n, k, method='random'):
    """
    Create the staring matrix given the row and columns count, using the method given in input
    
    Parameters:
        n (int): The row count of the matrix V
        k (int): The columns count the matrix V

    Returns:
        V (np.ndarray): The matrix V (nxk) initialisated with the selected method
    """
    
    if method == 'random':
        return np.random.rand(n, k)
    else:
        print('METHOD NOT FOUND, USING DEFAULT METHOD: RANDOM')
        return np.random.rand(n, k)


def start(A, k, test_name='test_name', data_folder='./data/test'):
    
    machine_precision = 1e-10

    
    liv_len = 2
    last_iteration_values = [x for x in range(liv_len)]
    max_iter = 10000
    
    m,n = A.shape
    start_time = None
    iteration_num = 0
    data_dict = {'obj_fun':[], 'U_norm':[], 'V_norm':[], 'iteration_time':[], 'iteration_id':[]}
    
    V_0 = get_starting_matrix(n, k)
    V_t = V_0.copy()
    norm_V_t = np.linalg.norm(V_t)
    
    while (abs(last_iteration_values[0] - last_iteration_values[-1])) > machine_precision and \
            max_iter > iteration_num:
        
        # start_time to measure execution time
        start_time = time.time()
        
        # computing U
        V_r, householder_vectors = thin_qr_factorization(V_t)
        AQ = apply_householder_transformations(A, householder_vectors)
        U_t = backward_substitution(AQ, np.transpose(V_r))

        # computing execution time
        exec_time_1 = time.time() - start_time
        
        # saving data of the iteration
        obj_fun_1 = objective_function(A,U_t,V_t) 
        norm_U_t = np.linalg.norm(U_t)
        
        data_dict['obj_fun'].append(obj_fun_1)
        data_dict['U_norm'].append(norm_U_t)
        data_dict['V_norm'].append(norm_V_t)
        
        data_dict['iteration_time'].append(exec_time_1)
        data_dict['iteration_id'].append(iteration_num)
        
        fancy_print(test_name, iteration_num, obj_fun_1, norm_U_t, norm_V_t, exec_time_1)
        
        
        # start_time to measure execution time
        start_time = time.time()
        
        # computing V
        U_r, householder_vectors = thin_qr_factorization(U_t)
        AQ = apply_householder_transformations(np.transpose(A), householder_vectors)
        V_t = backward_substitution(AQ, np.transpose(U_r))
        
        # computing execution time
        exec_time_2 = time.time() - start_time
        
        # saving data of the iteration
        obj_fun_2 = objective_function(A,U_t,V_t) 
        norm_V_t = np.linalg.norm(V_t)
        
        data_dict['obj_fun'].append(obj_fun_2)
        data_dict['U_norm'].append(norm_U_t)
        data_dict['V_norm'].append(norm_V_t)
        
        data_dict['iteration_time'].append(exec_time_2)
        data_dict['iteration_id'].append(iteration_num)
        
        fancy_print(test_name, iteration_num, obj_fun_2, norm_U_t, norm_V_t, exec_time_2)
        
        
        last_iteration_values.pop(0)
        last_iteration_values.append(obj_fun_2)
        iteration_num += 1
        
        
    save_data(A, U_t, V_0, V_t, data_dict, data_folder + '/' + test_name)

def fancy_print(name, iteration_num, obj_fun_1, norm_U, norm_V, exec_time):
    print(name + f' | it={iteration_num} | {exec_time} | obj={obj_fun_1} | U_norm={norm_U} | V_norm={norm_V} |')
    
def save_data(A, U, V_0, V, data_dict, directory):
    
    os.makedirs(directory, exist_ok=True)
    
    df = pd.DataFrame(data_dict)
    df.to_csv(directory + '/data.csv')
    
    np.save(directory + "/A.npy", A)
    np.save(directory + "/V_0.npy", V_0)
    np.save(directory + "/U.npy", U)
    np.save(directory + "/V.npy", V)