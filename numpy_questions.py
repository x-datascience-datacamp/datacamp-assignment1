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

    if X.type != "numpy.array":
        raise ValueError("Not a numpy.array")

    if np.shape(X) != 2 :
        raise ValueError("Not a 2D array")

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """
    my_pi = 1

    for i in range(1, n_terms+1) :
        my_pi *= 4*i**2/(4*i**2 - 1)

    return 2*my_pi
