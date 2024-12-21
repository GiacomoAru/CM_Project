import numpy as np
from scipy.linalg.lapack import dgeqrf, dorgqr
import pandas as pd
from datetime import datetime
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
    Q (ndarray):
    R (ndarray): The n x n upper triangular matrix.
    """
    Q, R = np.linalg.qr(A, mode='reduced')
    
    return Q, R
    

