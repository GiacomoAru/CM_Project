import numpy as np

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