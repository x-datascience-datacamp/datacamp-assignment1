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

    if type(X) != np.ndarray:
        raise ValueError("Input is not a Numpy Array")

    if X.ndim != 2:
        raise ValueError("Input is a Numpy array but must be 2D")

    n_samples, n_features = X.shape
    maxi = X[0][0]

    for row in range(n_samples):
        for col in range(n_features):
            if X[row, col] > maxi:
                i = row
                j = col
                maxi = X[row, col]

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    Parameters
    ----------
    n_terms : int
              number of terms in the product

    return
    ------
    res : int
          The approximation of pi/2 with n_terms in the product
    """

    res = 2.
    for n in range(1, n_terms+1):
        res *= 4*n**2/(4*n**2-1)

    return res
