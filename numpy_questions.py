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
        raise ValueError("Not an array")
    elif X.ndim != 2:
        raise ValueError("Not a 2D array")
    else:
        i, j = np.unravel_index(np.argmax(X, axis=None), X.shape)
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

      Parameters
    ----------
    n_terms: an int that corresponds to number of terms in the wallis product

    Returns
    -------
    pi: a float which is an approximation of pi.

    """
    prod = 1.0
    pi = 0
    if n_terms == 0:
        prod = 1.0
    for i in range(1, n_terms+1):
        term = 1 + 1 / (4 * i ** 2 - 1)
        prod *= term
    pi = 2 * prod
    return pi
