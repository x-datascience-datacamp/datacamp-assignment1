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

    # TODO
    if not isinstance(X, np.ndarray):
        raise ValueError("X should be a numpy array")
    if len(X.shape) != 2:
        raise ValueError("X should be a 2D array")
    ind = np.unravel_index(np.argmax(X), X.shape)
    i = ind[0]
    j = ind[1]

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    Parameters
    ----------
    n_terms : int
        The number of terms in the product. For example 10000.

    Returns
    -------
    estimated_pi : float
        The estimated value of pi using Wallis product

    Raises
    ------
    ValueError
        If the input is not an integer or
        if it's less than 0.

    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.
    if not isinstance(n_terms, int):
        raise ValueError("n_terms should be an integer")
    if n_terms < 0:
        raise ValueError("n_terms should be at least 0")

    estimated_pi = 1.0
    for n in range(1, n_terms+1):
        estimated_pi = estimated_pi * 4*n**2/(4*n**2-1)

    return 2*estimated_pi
