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

    if(isinstance(X, np.ndarray) and X.ndim == 2):
        i, j = np.unravel_index(np.argmax(X, axis=None), X.shape)
    else:
        raise ValueError("Error!")
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
    pi = 2
    if n_terms > 0:
        N = np.arange(1, n_terms+1)
        elem = (4 * N**2)/(4*N**2 - 1)
        pi *= np.product(elem)
    return pi
