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

    if not isinstance(X, np.ndarray):
        raise ValueError()
    if X.ndim != 2:
        raise ValueError()

    T = np.unravel_index(X.argmax(), X.shape)
    i = T[0]
    j = T[1]

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    """
    if not isinstance(n_terms, int):
        raise ValueError()

    nrange = np.arange(1, n_terms+1)
    wallis_product = (4 * nrange**2)/(4*nrange**2 - 1)
    result = np.product(wallis_product)

    return 2 * result
