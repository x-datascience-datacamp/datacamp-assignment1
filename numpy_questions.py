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

    if type(X).__module__ != np.__name__:
        raise ValueError("Input is not a numpy array")
    if len(X.shape) != 2:
        raise ValueError("Input is not 2D")
    i, j = np.unravel_index(np.argmax(X, axis=None), X.shape)
    return (i, j)


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    Parameters
    ----------
    n_terms : int
        The number of terms in the product

    Returns
    -------
    res : float
        The approximation of pi.

    """
    res = 1
    for term in range(1, n_terms+1):
        res *= (4*term**2)/(4*term**2-1)
    return 2*res
