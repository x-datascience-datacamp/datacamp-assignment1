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
    if isinstance(X, np.ndarray) and X.ndim == 2:
        idx = np.unravel_index(np.argmax(X), X.shape)
        i = idx[0]
        j = idx[1]
    else:
        raise ValueError('ValueError')

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    Parameters
    ----------
    n_terms : int
        The number of terms in the Wallis product.

    Returns
    -------
    wp : float
        The resulting Wallis product (doubled to get a pi approximation).

    Raises
    ------
    ValueError
        ####
    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.

    y = np.array([4 * x**2 / (4 * x**2 - 1) for x in range(1, n_terms+1)])
    z = np.prod(y)

    return 2 * z
