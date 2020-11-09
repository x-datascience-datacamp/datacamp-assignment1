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
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise ValueError

    i = 0
    j = 0
    i_max = np.argmax(X)
    i, j = np.unravel_index(i_max, X.shape)

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    Parameters
    --
    n_terms : positive integer
        number of terms to evaluates in the wallis product.

    Returns
    --
    approx_pi : floating point number
       approximation of pi given by the wallis product.
    """
    n = np.arange(1, n_terms + 1)
    terms = 4 * n**2
    terms = terms / (terms - 1)
    return 2 * np.multiply.reduce(terms, initial=1.0)
