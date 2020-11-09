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

    if not isinstance(X, np.ndarray):
        raise ValueError
    if X.ndim != 2:
        raise ValueError

    for k in range(X.shape[0]):
        for l in range(X.shape[1]):
            if X[k,l] > X[i,j]:
                i, j = k, l

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    """
    if not isinstance(n_terms, int):
        raise ValueError

    wallis_product = 2
    for i in range(1, n_terms + 1):
        wallis_product = wallis_product * 4 * i**2 / ( 4 * i**2 - 1)

    return wallis_product
