import numpy as np
import time
import pandas as pd
import os
import sys
import platform
import psutil
import socket
import uuid
from datetime import datetime
import json 

from off_the_shelf_utils import *

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
    X = np.zeros_like(A).astype('float64')
    
    eps = np.finfo(float).eps
    threshold = eps * np.max(np.abs(A))
    
    # for each column of X
    for i in range(k-1, -1, -1):
        # Solve for each row of X simultaneously
        if T[i,i] <=  threshold:
            X[:,i] = 0.0
        else:
            X[:,i] = (A[:,i] - np.sum( (T[i+1:,i] * X[:,i+1:]), 1)) * (1/T[i,i])

    return X

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

def get_starting_matrix(A, k, method='sketching_g', seed=None):
    """
    Create the starting matrix given the row and columns count, using the method given in input
    
    Parameters:
        A (np.ndarray): The input matrix whose size determines the dimensions of the starting matrix
        k (int): The number of columns in the matrix V
        method (str): The method for initialization ('rand_u', 'rand_n', 'scaled_u', 'scaled_n', 'sketching_g', 'sketching_b')
        seed (int, optional): The seed for random number generation (default is None, which means random seed)

    Returns:
        V (np.ndarray): The matrix V (n x k) initialized with the selected method
    """
    
    m, n = A.shape
    
    # Create a random number generator with the given seed (or default if None)
    rng = np.random.default_rng(seed)
    
    # uniform
    if method == 'rand_u':
        return rng.uniform(-1, 1, (n, k))
    
    # normal / gaussian distribution
    if method == 'rand_n':
        return rng.normal(0, 1, (n, k))
    
    # uniform but ||V|| = sqrt(||A||) -> ||V|| ~ ||U||
    if method == 'scaled_u':
        frob_A = np.linalg.norm(A, 'fro')
        max_val = np.sqrt(3 * np.sqrt(frob_A)**2 / (n * k))
        return rng.uniform(-max_val, max_val, (n, k))
    
    # normal but ||V|| = sqrt(||A||) -> ||V|| ~ ||U||
    if method == 'scaled_n':
        frob_A = np.linalg.norm(A, 'fro')  
        gamma = np.sqrt(frob_A) / np.sqrt(n * k)
        return rng.normal(0, gamma, (n, k))
    
    # Sketching with Gaussian distribution
    if method == 'sketching_g':
        
        V = np.transpose(rng.normal(0, 1, (k, m)) @ A)
        frob_A = np.linalg.norm(A, 'fro')
        frob_V = np.linalg.norm(V, 'fro')
        return (V/frob_V)*np.sqrt(frob_A*np.sqrt(k))
    
    # Sketching with Bernoulli distribution (-1, 1)
    if method == 'sketching_b':
        
        V = np.transpose(rng.choice([1, -1], size=(k, m)) @ A)
        frob_A = np.linalg.norm(A, 'fro')
        frob_V = np.linalg.norm(V, 'fro')
        return (V/frob_V)*np.sqrt(frob_A*np.sqrt(k))
    
    # Sketching with Gaussian distribution
    if method == 'semi-orthogonal':
        
        R, h_v = thin_qr_factorization(rng.normal(0, 1, (n, k)))
        return apply_householder_transformations(np.eye(n), h_v)
    
    else:
        print('METHOD NOT FOUND, USING DEFAULT METHOD: rand_n')
        return rng.normal(0, 1, (n, k))


def start(A, k, c_name='class', m_name='matrix', t_name='test', init_method='snormal', data_folder='./data/test', 
          max_iter = 20000, liv_len = 2, epsilon = np.finfo(np.float64).eps, seed=None):
    
    if liv_len < 2:
        liv_len = 2
    if max_iter < 1:
        max_iter = 1   
    if epsilon < np.finfo(np.float64).eps:
        epsilon = np.finfo(np.float64).eps
        
    # Get current date and time
    input_values = {
        'k':k,
        'c_name': c_name,
        'm_name': m_name,
        't_name': t_name,
        'init_method': init_method,
        'data_folder': data_folder,
        'max_iter': max_iter,
        'liv_len': liv_len,
        'epsilon': epsilon,
        'seed':seed,
        'date':datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # stop handler
    last_iteration_values = [x + 1 for x in range(liv_len)]
    
    start_time = qr_time = manip_time = bw_time = None
    iteration_num = 0
    data_dict = {'obj_fun':[], # 'UV_norm':[], 
                 'U_norm':[], 'V_norm':[], 
                 'qr_time':[], 'manip_time':[], 'bw_time':[], 
                 'iteration_id':[]}
    
    V_0 = get_starting_matrix(A, k, init_method, seed)
    V_t = V_0.copy()
    norm_V_t = np.linalg.norm(V_t)
    
    # iterate until convergence or until a maximum number of iterations is reached
    while (np.abs(last_iteration_values[0] - last_iteration_values[-1])) > epsilon * max(np.abs(last_iteration_values[0]), np.abs(last_iteration_values[-1])) \
        and max_iter > iteration_num:
        
        
        # computing U
        start_time = time.time()
        V_r, householder_vectors = thin_qr_factorization(V_t)
        print('V_r', V_r)
        qr_time = time.time() - start_time
        start_time = time.time()
        AQ = apply_householder_transformations(A, householder_vectors)
        print('AQ', AQ)
        manip_time = time.time() - start_time
        start_time = time.time()
        U_t = backward_substitution(AQ, np.transpose(V_r))
        print('U_t', U_t)
        bw_time = time.time() - start_time
    
        
        # saving data of the iteration
        UV = np.dot(U_t, np.transpose(V_t))
        
        # norm_UV = np.linalg.norm(UV, 'fro')
        obj_fun = np.linalg.norm(A - UV, 'fro')
        norm_U_t = np.linalg.norm(U_t)
        
        data_dict['obj_fun'].append(obj_fun)
        # data_dict['UV_norm'].append(norm_UV)
        data_dict['U_norm'].append(norm_U_t)
        data_dict['V_norm'].append(norm_V_t)
        
        data_dict['qr_time'].append(qr_time)
        data_dict['manip_time'].append(manip_time)
        data_dict['bw_time'].append(bw_time)
        data_dict['iteration_id'].append(iteration_num)
        
        _fancy_print(m_name, t_name, iteration_num, obj_fun, norm_U_t, norm_V_t, qr_time+manip_time+bw_time)
        
        input()

        # computing V
        start_time = time.time()
        U_r, householder_vectors = thin_qr_factorization(U_t)
        print('U_r', U_r)
        qr_time = time.time() - start_time
        start_time = time.time()
        AQ = apply_householder_transformations(np.transpose(A), householder_vectors)
        print('AQ', AQ)
        manip_time = time.time() - start_time
        start_time = time.time()
        V_t = backward_substitution(AQ, np.transpose(U_r))
        print('V_t', V_t)
        bw_time = time.time() - start_time


        # saving data of the iteration
        UV = np.dot(U_t, np.transpose(V_t))
        
        # norm_UV = np.linalg.norm(UV, 'fro')
        obj_fun = np.linalg.norm(A - UV, 'fro')
        norm_V_t = np.linalg.norm(V_t)
        
        data_dict['obj_fun'].append(obj_fun)
        # data_dict['UV_norm'].append(norm_UV)
        data_dict['U_norm'].append(norm_U_t)
        data_dict['V_norm'].append(norm_V_t)
        
        data_dict['qr_time'].append(qr_time)
        data_dict['manip_time'].append(manip_time)
        data_dict['bw_time'].append(bw_time)
        data_dict['iteration_id'].append(iteration_num)
        
        _fancy_print(m_name, t_name, iteration_num, obj_fun, norm_U_t, norm_V_t, qr_time+manip_time+bw_time)
        
        #print(np.max(A - UV))
        input()
        
        last_iteration_values.pop(0)
        last_iteration_values.append(obj_fun)
        iteration_num += 1
        
        
    _save_data(input_values, A, U_t, V_0, V_t, data_dict, data_folder, c_name, m_name, t_name)

def _fancy_print(m_name, t_name, iteration_num, obj_fun, norm_U, norm_V, exec_time):
    print(f'{m_name} | {t_name} | {iteration_num} | {exec_time:.3f}s | obj={obj_fun:.17f} | U={norm_U:.17f} | V={norm_V:.17f} |')

def _full_pc_info():
    # Operating System and Version
    os_name = platform.system()
    os_version = platform.version()
    pc_name = platform.node()
    architecture = platform.architecture()[0]
    machine = platform.machine()
    processor = platform.processor()

    # CPU Details
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    cpu_frequency = psutil.cpu_freq().current if hasattr(psutil, 'cpu_freq') else 'N/A'
    
    # Memory
    total_memory = psutil.virtual_memory().total / (1024 ** 3)  # in GB
    available_memory = psutil.virtual_memory().available / (1024 ** 3)  # in GB

    # Disk
    total_disk = psutil.disk_usage('/').total / (1024 ** 3)  # in GB
    free_disk = psutil.disk_usage('/').free / (1024 ** 3)  # in GB

    # Unique Machine ID (MAC Address)
    unique_id = str(uuid.getnode())  # MAC address, often used as unique identifier

    # Storage System
    storage_system = os.name  # 'posix' for Linux/Mac, 'nt' for Windows

    # Creating a dictionary to hold all the information
    data = {
        'operating_system': f"{os_name} {os_version}",
        'pc_name': pc_name,
        'architecture': architecture,
        'machine': machine,
        'processor': processor,
        'physical_cores': physical_cores,
        'logical_cores': logical_cores,
        'cpu_frequency_mhz': cpu_frequency,
        'total_memory_gb': total_memory,
        'available_memory_gb': available_memory,
        'total_disk_space_gb': total_disk,
        'free_disk_space_gb': free_disk,
        'unique_machine_id_mac_address': unique_id,
        'storage_system': storage_system
    }
    
    return data

def _save_data(input_values, A, U, V_0, V, data_dict, data_folder, c_name, m_name, t_name):
    
    directory = data_folder + '/' + c_name + '/' + m_name + '/' + t_name
        
    os.makedirs(directory, exist_ok=True)
    
    df = pd.DataFrame(data_dict)
    df.to_csv(directory + '/data.csv', index=False)
    
    np.save(directory + "/A.npy", A)
    np.save(directory + "/V_0.npy", V_0)
    np.save(directory + "/U.npy", U)
    np.save(directory + "/V.npy", V)
    
    with open(directory + "/input_values.json", 'w+') as f:
        f.write(json.dumps(input_values))
    with open(directory + "/pc_info.json", 'w+') as f:
        f.write(json.dumps(_full_pc_info()))



def test_start(A, k, c_name='class', m_name='matrix', t_name='test', init_method='snormal', data_folder='./data/test', 
          max_iter = 20000, liv_len = 2, epsilon = np.finfo(np.float64).eps, seed=None):
    
    if liv_len < 2:
        liv_len = 2
    if max_iter < 1:
        max_iter = 1   
    if epsilon < np.finfo(np.float64).eps:
        epsilon = np.finfo(np.float64).eps
        
    # Get current date and time
    input_values = {
        'k':k,
        'c_name': c_name,
        'm_name': m_name,
        't_name': t_name,
        'init_method': init_method,
        'data_folder': data_folder,
        'max_iter': max_iter,
        'liv_len': liv_len,
        'epsilon': epsilon,
        'seed':seed,
        'date':datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # stop handler
    last_iteration_values = [x + 1 for x in range(liv_len)]
    
    start_time = qr_time = manip_time = bw_time = None
    iteration_num = 0
    data_dict = {'obj_fun':[], # 'UV_norm':[], 
                 'U_norm':[], 'V_norm':[], 
                 'qr_time':[], 'manip_time':[], 'bw_time':[], 
                 'iteration_id':[]}
    
    V_0 = get_starting_matrix(A, k, init_method, seed)
    V_t = V_0.copy()
    norm_V_t = np.linalg.norm(V_t)
    
    # iterate until convergence or until a maximum number of iterations is reached
    while (np.abs(last_iteration_values[0] - last_iteration_values[-1])) > epsilon * max(np.abs(last_iteration_values[0]), np.abs(last_iteration_values[-1])) \
        and max_iter > iteration_num:
        
        
        # computing U
        start_time = time.time()
        V_r, householder_vectors = thin_qr_factorization_OTS(V_t)
        qr_time = time.time() - start_time
        start_time = time.time()
        AQ = apply_householder_transformations(A, householder_vectors)
        manip_time = time.time() - start_time
        start_time = time.time()
        U_t = backward_substitution(AQ, np.transpose(V_r))
        bw_time = time.time() - start_time
    
        
        # saving data of the iteration
        UV = np.dot(U_t, np.transpose(V_t))
        
        # norm_UV = np.linalg.norm(UV, 'fro')
        obj_fun = np.linalg.norm(A - UV, 'fro')
        norm_U_t = np.linalg.norm(U_t)
        
        data_dict['obj_fun'].append(obj_fun)
        # data_dict['UV_norm'].append(norm_UV)
        data_dict['U_norm'].append(norm_U_t)
        data_dict['V_norm'].append(norm_V_t)
        
        data_dict['qr_time'].append(qr_time)
        data_dict['manip_time'].append(manip_time)
        data_dict['bw_time'].append(bw_time)
        data_dict['iteration_id'].append(iteration_num)
        
        _fancy_print(m_name, t_name, iteration_num, obj_fun, norm_U_t, norm_V_t, qr_time+manip_time+bw_time)



        # computing V
        start_time = time.time()
        U_r, householder_vectors = thin_qr_factorization_OTS(U_t)
        qr_time = time.time() - start_time
        start_time = time.time()
        AQ = apply_householder_transformations(np.transpose(A), householder_vectors)
        manip_time = time.time() - start_time
        start_time = time.time()
        V_t = backward_substitution(AQ, np.transpose(U_r))
        bw_time = time.time() - start_time


        # saving data of the iteration
        UV = np.dot(U_t, np.transpose(V_t))
        
        # norm_UV = np.linalg.norm(UV, 'fro')
        obj_fun = np.linalg.norm(A - UV, 'fro')
        norm_V_t = np.linalg.norm(V_t)
        
        data_dict['obj_fun'].append(obj_fun)
        # data_dict['UV_norm'].append(norm_UV)
        data_dict['U_norm'].append(norm_U_t)
        data_dict['V_norm'].append(norm_V_t)
        
        data_dict['qr_time'].append(qr_time)
        data_dict['manip_time'].append(manip_time)
        data_dict['bw_time'].append(bw_time)
        data_dict['iteration_id'].append(iteration_num)
        
        _fancy_print(m_name, t_name, iteration_num, obj_fun, norm_U_t, norm_V_t, qr_time+manip_time+bw_time)
        
        #print(np.max(A - UV))
        #time.sleep(10)
        
        last_iteration_values.pop(0)
        last_iteration_values.append(obj_fun)
        iteration_num += 1
        
        
    _save_data(input_values, A, U_t, V_0, V_t, data_dict, data_folder, c_name, m_name, t_name)