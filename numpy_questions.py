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

    i = X.argmax(0)
    j = X.argmax(1)
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a np.array")
    if type(X) != 'numpy.ndarray': 
        raise ValueError(": not a 2D array")
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

     Parameters
    ----------
    n_terms : number of product in Wallis product 

    Returns
    -------
    pi : approximation of pi following the Wallis product 


    Raises
    ------
    ValueError
        If the input is not an int
    """

    if n_terms > 0:
        N = np.arange(1, n_terms+1)
        elem = (4 * N**2)/(4*N**2 - 1)
        pi *= np.product(elem)
        pi *= 2
    return pi
