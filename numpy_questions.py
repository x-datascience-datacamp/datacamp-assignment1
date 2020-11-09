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
    try:
        i, j = np.unravel_index(np.argmax(X, axis=None), X.shape)
        return i, j
    except Exception:
        raise ValueError("Not a numpy array")



def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

   Parameters
    ----------
    n_terms : int
              Indicates the number of terms to compute.

    Returns
    -------
    pi : float
         approximation of pi

    """

    pi = 2.
    for i in range(1, n_terms + 1):
        left = (2. * i)/(2. * i - 1.)
        right = (2. * i)/(2. * i + 1.)
        pi = pi * left * right
    return pi

