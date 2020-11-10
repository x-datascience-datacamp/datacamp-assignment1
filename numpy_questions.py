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
    k = 0
    h = 0

    if (type(X) != np.ndarray) or (X.ndim != 2):
        raise ValueError

    n, m = np.shape(X)
    max = X[0][0]

    for i in range(n):
        for j in range(m):
            if X[i][j] > max:
                k, h = i, j
                max = X[i][j]
    return k, h


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.

    p = 1
    for i in range(1, n_terms+1):
        p = p * (4*i**2 / (4*i**2 - 1))

    return 2*p
