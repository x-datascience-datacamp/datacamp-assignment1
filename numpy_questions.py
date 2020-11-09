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
        raise ValueError("X is an numpy array")
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    i, j = np.unravel_index(np.argmax(X, axis=None), X.shape)
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    Parameters
    ----------
    n_terms : integer
        Number of terms in the product.

    Returns
    -------
    pi : float
        The approximation of pi.

    Raises
    ------
    ValueError
        If n_terms < 0.
    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.
    if n_terms < 0:
        raise ValueError("Number of terms must be higher than 0")
    pi = 1.
    for i in range(1, n_terms+1):
        pi *= (4*(i**2)) / ((4*(i**2)-1))
    return pi*2
