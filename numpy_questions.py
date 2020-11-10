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

    if(type(X) is not np.ndarray):
        raise ValueError("X isn't a numpy array")
    elif(len(X.shape) != 2):
        raise ValueError("the array shape is not 2D")
    else:
        i = np.argmax(np.max(X, axis=1))
        j = np.argmax(np.max(X, axis=0))
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    Parameters
    ----------
    n_terms : integer
        The n_terms is an int that corresponds to the number of terms
        in the product. For example 10000.

    Returns
    -------
    pi : float
        pi value approximation

    """
    pi_2 = 1.
    pi_n = 0.

    for n in range(1, n_terms+1):
        pi_n = (4.0 * n**2)/(4.0 * n**2 - 1.0)
        pi_2 = pi_2 * pi_n
    pi = pi_2 * 2

    return pi
