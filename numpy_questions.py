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

    if type(X).__module__ == np.__name__:
        raise ValueError("X is not an numpy array")
    if X.ndim != 2:
        raise ValueError("X shape is not 2D")

    max = X[0][0]
    for x in range(0, X.shape[0]):
        for y in range(0, X.shape[1]):
            if X[x][y] > max :
                i = x
                j = y

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.

    product = 1
    for n in range(n_terms) :
        product *= (4*n**2) / (4*n**2 - 1)

    return 0.
