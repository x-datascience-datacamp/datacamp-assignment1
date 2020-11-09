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

    if type(X) is not np.ndarray:
        raise ValueError("the input is not a numpy array")
    if len(X.shape) != 2:
        raise ValueError("the shape is not 2D")
    maxi = X.max()
    maxi_i_j = np.where(X == maxi)
    i = maxi_i_j[0][0]
    j = maxi_i_j[1][0]
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.
    See:
    https://en.wikipedia.org/wiki/Wallis_product
    Parameters
    ----------
    n_terms : the number of terms for the approximation of pi.
    Returns
    -------
    pi_approx : the approximate pi value with wallis product (n_terms times).
    Raises
    ------
    ValueError
        If n_terms < 0
    """
    pi_approx = 2.0
    if n_terms < 0:
        raise ValueError("n_terms < 0")

    if n_terms == 0:
        return pi_approx

    for i in range(1, n_terms+1):
        pi_approx *= ((4*i**2)/(4*i**2-1))
    return pi_approx
