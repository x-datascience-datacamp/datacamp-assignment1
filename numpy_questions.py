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

    if isinstance(X, np.ndarray) and np.ndim(X) == 2:
        i, j = np.unravel_index(np.argmax(X), np.shape(X))
    else:
        raise ValueError

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    Parameters
    ----------
    n_terms : number of terms in the Wallis product

    Returns
    -------
    x : approximated value of Pi
    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.
    x = 2
    for i in range(n_terms):
        x = x*(4*(i+1)**2)/((4*(i+1)**2)-1)
    return x
