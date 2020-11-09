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
    # Raises
    if not isinstance(X, np.ndarray):
        raise ValueError("the input is not numpy")
    if not (X.ndim == 2):
        raise ValueError("the shape is not 2D.")
    # TODO
    idx = np.argmax(X)
    i = idx // X.shape[1]
    j = idx - X.shape[1] * i
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """
    pi = 1

    for i in range(n_terms):
        pi = 1. * pi * 4*(i+1)**2 / (4*(i+1)**2-1)

    return pi*2

    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.\
