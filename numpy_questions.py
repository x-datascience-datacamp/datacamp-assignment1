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
    if type(X) is not np.ndarray:
        raise ValueError('The input is not a np array')
    if len(X.shape) != 2:
        raise ValueError('X is not a np array')
    (i, j) = np.unravel_index(np.argmax(X), X.shape)
    return(i, j)


def wallis_product(n_terms):
    pi = 1
    for i in range(1, n_terms+1):
        pi *= 4*i**2 / (4 * i**2 - 1)
    pi *= 2
    return pi
