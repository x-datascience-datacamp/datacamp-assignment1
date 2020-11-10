"""This is the numpy test."""

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

    if X is None or type(X) is not np.ndarray or X.ndim != 2:
        raise ValueError("Value Error")
    i, j = np.where(X == np.max(X))

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    Args:
        n_terms: n_terms used to approximation.

    Returns:
        pi: Pi approximation.
    See: https://en.wikipedia.org/wiki/Wallis_product]

    """
    pi = 1.0
    for i in range(1, n_terms+1):
        pi *= (4*i**2)/(4*i**2-1)

    pi = pi * 2

    return pi
