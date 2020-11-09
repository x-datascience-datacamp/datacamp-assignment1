clear()
import numpy as np
def max_index(X):
    i = 0
    j = 0
    if type(X) is not np.ndarray :
        raise ValueError('The input is not a np array')
    if len(X.shape) !=2 : 
        raise ValueError('X is not a matrix')
    (i, j) = np.unravel_index(np.argmax(X), X.shape) 
    return (i, j)

def wallis_product(n_terms):
    X = np.arange(1, n_terms+1, dtype=float)
    X = 4 * X**2 / (4 * X**2 - 1)
    pi = 2 * np.product(X)
    return pi
    
