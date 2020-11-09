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

    # TODO
    if(X is None or type(X) is not np.ndarray):
        raise ValueError("ValueError exception thrown")
    else:
        i, j = np.unravel_index(np.argmax(X, axis=None), X.shape)
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    Parameters
    ----------
    n_terms :   int
                number of terms in the product. For example 10000.

    Returns
    -------
    product :   float
                pi value.

    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.
    if(n_terms > 0):
        product = 4*np.linspace(1, n_terms, n_terms)**2
        product = product/(product-1)
        product = product.prod()
    else:
        product = 1
    return 2*product
