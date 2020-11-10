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

    if type(X) != np.ndarray:
        raise ValueError()
    if len(X.shape) != 2:
        raise ValueError()

    n, d = X.shape
    a = np.argmax(X)
    i = a // d
    j = a % d
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    Parameters
    ----------
    n_terms : int
        The number of terms in the product

    Returns
    -------
    pi_approx : int
        An approximation of the number PI
    """

    A = np.array([4*n**2/(4*n**2-1) for n in range(1, n_terms+1)])
    pi_approx = 2*A.prod(axis=0)
    return pi_approx
