
# noqa: D100
import numpy as np


def max_index(X):  # noqa: D202
    """Return the index of the maximum in a numpy array.

    Args
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input array
    Returns
    -------
    i : int
        The row index of the maximum.
    j : int
        The column index of the maximum.
    Raises
    ------
    ValueError
        If the input is not a numpy error or if the shape is not 2D.
    """

    i = 0
    j = 0
    if (type(X) is not np.ndarray):
        raise ValueError("Numpy error")
    if (type(X) is None):
        raise ValueError("None")
    if (len(X.shape) != 2):
        raise ValueError("Not 2D Shape")
    i, j = np.unravel_index(X.argmax(), np.shape(X))
    return i, j


def wallis_product(n_terms):
    """
    Implement the Wallis product to compute an approximation of pi.
    """

    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.

    pi = 1
    if n_terms == 1:
        pi = 4 / (4-1)
    for i in range(1, n_terms):
        pi *= float(4 * i ** 2)/float(4 * i ** 2 - 1)
    pi *= 2
    return pi
