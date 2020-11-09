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

    if not(isinstance(X, np.ndarray)):
        raise ValueError("Input is not a numpy array!")
    elif (X.ndim != 2):
        raise ValueError("Input is numpy array but its dimnesion is " +
                   str(X.ndim)+". Required dimension is 2.")
    else :
        for idx1 in range(X.shape[0]) :
            for idx2 in range(X.shape[1]) :
                if X[idx1,idx2] > X[i,j] :
                    i,j = idx1,idx2  

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.
    pi = 2
    for i in range(1,n_terms+1):
        pi *= (4*i**2)/(4*i**2-1)
    return pi

