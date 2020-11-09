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

    # TODO
    if not isinstance(X, np.ndarray):
        raise ValueError("X not None")
    if X.ndim != 2:
        raise ValueError("X must 2D array")
    Max = np.max(X)
    argmax = np.where(X == Max)
    i, j = argmax[0][0], argmax[1][0]
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """
    pi = 1
    for i in range(1, n_terms+1):
        pi *= (4*(np.square(i)))/(4*(np.square(i))-1)
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.
    return 2*pi
