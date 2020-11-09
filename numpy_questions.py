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
    # Raise errors
    if type(X) is not np.ndarray:
        raise ValueError('X should be a ndarray')
    if len(X.shape) != 2:
        raise ValueError('X should be 2D')

    i, j = np.unravel_index(X.argmax(), X.shape)

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    Parameters
    ----------
    X : int
        The number of terms in the product.

    Returns
    -------
    pi : int
         The product that approximates pi.
    """
    pi = 2
    for i in range(1, n_terms+1):
        pi = pi * ((2*i)/(2*i-1)) * ((2*i)/(2*i+1))

    return pi
