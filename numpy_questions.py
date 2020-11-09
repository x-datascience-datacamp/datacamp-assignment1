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
    i = 0
    j = 0

    if not isinstance(X, np.ndarray):
        raise ValueError("The input is not a numpy array.")

    if len(X.shape) != 2:
        raise ValueError("The shape is not 2D.")

    n_samples, n_features = X.shape

    for k in range(n_samples):
        for m in range(n_features):
            if X[k, m] > X[i, j]:
                i, j = k, m

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    Parameters
    ----------
    n_terms : int
              The number of terms of the Wallis product to compute.

    Returns
    -------
    prod : float
           The Wallis product with n_terms factors.

    Raises
    ------
    ValueError
        If the input is not an int.
    """
    if not isinstance(n_terms, int):
        raise ValueError("The input has to be an int.")

    prod = 1

    for k in range(1, n_terms+1):
        prod *= 4 * k**2 / (2 * k - 1) / (2 * k + 1)

    return 2 * prod
