# noqa: D100
"""Functions for numpy trials."""
import numpy as np


def max_index(X):
    """
    Return the index of the maximum in a numpy array.

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
    if isinstance(X, np.ndarray):
        shape = X.shape
        if len(shape) == 2:
            ind = np.unravel_index(np.argmax(X), shape)
            i, j = ind[0], ind[1]
            return i, j
        else:
            raise(ValueError)
    else:
        raise(ValueError)


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    Parameters
    ----------
    n_terms : int32
        The number of terms in the product.

    Returns
    -------
    res : float
        The estimated value of Pi.

    Raises
    ------
    ValueError
        If the input is not an int or
        If the input is negative

    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.

    if not isinstance(n_terms, int) or (n_terms < 0):
        raise(ValueError)
    res = 2.
    for i in range(n_terms):
        res *= 4 * (i+1) ** 2 / (4 * (i+1) ** 2 - 1)

    return res
