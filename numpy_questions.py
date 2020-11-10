# noqa: D100
import numpy as np


def max_index(X):
    """Return the index of the maximum in a numpy array.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features) The input array.
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
        raise ValueError('the input is not a numpy')
    # TODO
    if X.ndim != 2:
        raise ValueError('the shape is not 2D')
    i, j = np.where(X == np.max(X))
    i = i[0]
    j = j[0]
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """
    pi_approx = 2
    if n_terms > 0:
        i = np.arange(1, n_terms+1)
        first_term = (2. * i)/(2. * i - 1.)
        secont_term = (2. * i)/(2. * i + 1.)
        pi_approx *= np.product(first_term*secont_term)
    return pi_approx
