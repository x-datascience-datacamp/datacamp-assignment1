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
        If the input is not a numpy array or
        if the shape is not 2D.
    """
    i = 0
    j = 0

    if type(X) != np.ndarray:
        raise ValueError("The input must be a numpy array")
    if len(X.shape) != 2:
        raise ValueError("The input must be a 2D array")

    i = np.argmax(np.max(X, axis=1))
    j = np.argmax(np.max(X, axis=0))

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    Parameters
    ----------
    n_terms : int
        Number of terms in the Wallis product.

    Returns
    -------
    pi : float
        Approximation of pi.
    """
    # The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.

    pi = 2
    for i in range(1, n_terms+1):
        pi = pi * (1 + 1/(4*i**2 - 1))

    return pi
