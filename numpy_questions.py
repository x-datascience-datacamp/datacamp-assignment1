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
    if X is None or not isinstance(X, np.ndarray):
        raise ValueError("The input array X is None or not a numpy array.")
    if X.ndim != 2:
        raise ValueError("The input array X is not 2D.")
    # maximum value
    max_value = X.max()
    max_indices = np.where(X == max_value)
    i, j = max_indices[0][0], max_indices[1][0]
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    Parameters
    ----------
    n_terms     : int
                  the number of terms to compute the approximate of PI

    Returns
    -------
    pi_estimate : float
                  the approximate value of pi after n iterations of the wallis
                  product

    Raises
    ------
    ValueError
        If n_terms is negative
    """
    if n_terms < 0:
        raise ValueError("The number of terms to compute pi is negative")
    pi_estimate = 2.
    # initial term of the serie
    if n_terms == 0:
        return pi_estimate
    # apply the series formula
    n = np.arange(1, n_terms+1)
    wallis_product = (4 * n**2)/(4*n**2 - 1)
    return pi_estimate * np.product(wallis_product)
