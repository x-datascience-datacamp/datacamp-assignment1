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
    try:
        X.ndim == 2
    except:
        raise ValueError
    i, j = np.unravel_index(X.argmax(), X.shape)

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    Parameters
    ----------
    n_terms : int
        Number of terms used for the approximation

    Returns
    -------
    pi_approx: float
        The approximation of pi computed

    Raises
    ------
    ValueError
        If the input is not a numpy error or
        if the shape is not 2D.
    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.
    X = 4 * np.arange(1, n_terms + 1) ** 2
    X = X / (X - 1)
    pi_approx = 2 * X.prod()
    print(pi_approx)
    return pi_approx

wallis_product(1)