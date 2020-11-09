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
    if type(X) is not np.ndarray or X is None:
        raise ValueError("Input should be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input should have 2 dimensions")
    i, j = np.unravel_index(np.argmax(X), X.shape)
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product


    Parameters
    ----------
    n_terms : number of terms to consider in computing
              the wallis product

    Returns
    -------
    value   : the wallis product, an approximation of
               pi

    Raises
    ------
    ValueError
        If n_terms is not an integer or
        if n_terms is less than 1
    """
    if type(n_terms) != int or n_terms < 0:
        raise ValueError("The number of terms should be a positive integer")
    value = 1
    for i in range(n_terms):
        value *= (4*(i+1)**2)/(4*(i+1)**2-1)
    return 2*value
