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
    if X.ndim != 2:
        raise ValueError(" The input shape of X should be 2D")
    L, C = np.shape(X)
    for line in range(0, L):
        for colonne in range(0, C):
            if X[line][colonne] > X[i][j]:
                i = line
                j = colonne
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.
    pi = 2
    if n_terms > 0:
        for n in range(1, n_terms+1):
            pi *= (4*n**2)/(4*n**2 - 1)
    return pi
