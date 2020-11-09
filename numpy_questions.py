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
    if type(X) is not np.ndarray:
        raise ValueError("The input X is not a numpy array ")
    if len(X.shape) != 2:
        raise ValueError("The input X is not 2D")
    M = X.max()  # MAX value of the matrix
    i, j = np.where(X == M)
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    Parameters
    ----------------------
    n_term : the number

    Returns
    ----------------------
    x : the approximation of pi

    Raises
    ----------------------
    Error if n_term is not or negative
    """
    if n_terms is None:
        raise ValueError("The input in None")
    if (n_terms < 0):
        raise ValueError("The input should be positive")
    x = 1
    if n_terms == 0:
        return 2.*x
    for i in range(1, n_terms + 1):
        x *= ((4*i**2) / (4*i**2 - 1))
    return (2.*x)
