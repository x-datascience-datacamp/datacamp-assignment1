# noqa: D100
import numpy as np
from numpy import unravel_index


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

    if X is None or type(X) is list or X.ndim < 2:

        raise ValueError("ValueError exception thrown")

    else:

        i = unravel_index(X.argmax(), X.shape)[0]
        j = unravel_index(X.argmax(), X.shape)[1]

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    Parameters
    ----------
    n_terms : int
        Number of terms oin the Wallis product.

    Returns
    -------
    v : float
        Computed Wallis product.

    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.

    v = [(2*k/(2*k-1))*(2*k/(2*k+1)) for k in range(1, n_terms+1)]
    r = 2*np.prod(v)

    return r
