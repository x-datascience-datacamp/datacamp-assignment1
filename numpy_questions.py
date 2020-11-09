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
        raise ValueError('Error in type or shape')
    (i, j) = np.unravel_index(np.argmax(X, axis=None), X.shape)
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.
    Parameters
    ----------
    n_terms : int, The number of terms in the product

    Returns
    -------
    pi : float
         Approximation of Pi by n_terms Wallis formula
    """
    pi = 2.

    for i in range(1, n_terms + 1):
        element_1 = (2. * i)/(2. * i - 1.)
        element_2 = (2. * i)/(2. * i + 1.)
        pi = pi * element_1 * element_2

    return pi
