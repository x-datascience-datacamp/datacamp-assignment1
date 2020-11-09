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
        If the input is not a numpy array or
        if the shape is not 2D.
    """
    i = 0
    j = 0

    if not isinstance(X, np.ndarray):
        raise ValueError("Wrong type")
    if X.ndim != 2:
        raise ValueError("Wrong dimension")

    maximum = np.max(X)
    i, j = np.where(X == maximum)[0][0], np.where(X == maximum)[1][0]

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.

    resu = 2.

    n = np.arange(1, n_terms+1)
    terms_to_multiply = (2*n / (2*n-1)) * (2*n / (2*n+1))
    resu *= np.product(terms_to_multiply)

    return resu
