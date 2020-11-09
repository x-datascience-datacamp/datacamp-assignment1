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
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a np.array")
    if X.ndim != 2:
        raise ValueError("Input shape must be 2D")
    i, j = np.where(X == np.max(X))
    i = i[0]
    j = j[0]
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    Parameters
    ----------
    n_terms : int
        number for terms in the product. Higher is more precise.

    Returns
    -------
    pi : float
        approximation of pi.

    """
    pi = 2
    if n_terms > 0:
        n = np.arange(1, n_terms+1)
        product_terms = (4 * n**2)/(4*n**2 - 1)
        pi *= np.product(product_terms)
    return pi
