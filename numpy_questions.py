# noqa: D100py
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

    if not (type(X) == np.ndarray or type(X) == np.matrix):
        raise ValueError("Input data must be a Numpy array or matrix")
    if not (X.ndim==2):
        raise ValueError("Input data must be 2D")
    

    T = np.unravel_index(X.argmax(), X.shape)
    i=T[0]
    j=T[1]

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """
    pi=1.0
    for j in range(1, n_terms):
        pi = pi* 4 * j ** 2 / (4 * j ** 2 - 1)


    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.
    return 2*pi
