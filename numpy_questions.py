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
    try:
        i, j = np.unravel_index(np.argmax(X, axis=None), X.shape)
        return i, j
    except Exception:
        raise ValueError("Not a numpy array")


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    Parameters
    ----------
    n_terms     : int
                  the number of terms to compute the approximate of PI
    Returns
    -------
    pi : float
                  the approximate value of pi after n iterations of the wallis
                  product
    Raises
    ------
    ValueError
        If n_terms is negative
    """
    pi = 2.
    if n_terms == 0:
        return pi
    if n_terms < 0:
        raise ValueError("n_terms is negative")
    n = np.arange(1, n_terms+1)
    wallis_product = (4 * n**2)/(4*n**2 - 1)
    return pi * np.product(wallis_product)
