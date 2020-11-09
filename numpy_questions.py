# noqa: D100
import numpy as np


def max_index(X):
    """Return the index of the maximum in a numpy array.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input array.

    Returns
    -------
    i : int
        The row index of the maximum.

    j : int
        The column index of the maximum.

    Raises
    ------
    ValueError
        If the input is not a numpy error or
        if the shape is not 2D.
    """
    i = 0
    j = 0

    # TODO
    max = X[0,0]
    index_i = 0
    index_j = 0
    for i in range X.shape[0]:
        for j in range X.shape[1]:
            if X[i,j] > max:
                index_i = i
                index_j = j
                max = X[i,j]
                
    i = index_i
    j = index_j

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.
    list_1 = np.arange(1,n_terms+1)
    prod = 1
    for i in range (2,n_terms+1,step=2):
        prod = prod*((i/(i-1))*(i/i+1)) 
    return 0.
