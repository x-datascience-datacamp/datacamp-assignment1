# noqa: D100
import numpy as np



    


def max_index(X):
    
    i = 0
    j = 0
    # TODO
    if type(X) is not np.ndarray:
         raise ValueError('The input is not a np array')
    if len(X.shape) !=2 :
         raise ValueError('X is not a matrix')
    for l in range(X.shape[0]):
         for k in range(X.shape[1]):
             if X[l,k] > X[i,j]:
                 i, j = l, k
     for line in range(X.shape[0]):
         for col in range(X.shape[1]):
             if X[line, col] > X[i, j]:
                 i, j = line, col
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.
    See:
    https://en.wikipedia.org/wiki/Wallis_product
    XXX : write Parameters and Returns sections as above.
    """
    
    X = np.arange(1, n_terms+1, dtype=float)
    X = 4 * X**2 / (4 * X**2 - 1)
    pi = 2 * np.product(X)
    
    return pi
    
