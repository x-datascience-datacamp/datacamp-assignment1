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
    if isinstance(X, np.ndarray) and len(X.shape) == 2:
        i, j = np.unravel_index(X.argmax(), X.shape)
    else:
        raise ValueError
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    Parameters
    ----------
    n_terms : int
        Number of terms in Wallis product.

    Returns
    -------
    wallis_p : float
        Approximation of pi computed using Wallis product

    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.
    assert isinstance(n_terms, int)

    wallis_p = 1.
    if n_terms > 0:
        for n in range(1, n_terms + 1):
            value_n = 4 * n**2
            wallis_p = wallis_p * value_n / (value_n - 1)
    return 2 * wallis_p
