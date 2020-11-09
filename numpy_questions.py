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
    if not isinstance(X, np.ndarray) or not len(X.shape) == 2:
        raise ValueError("X should be a 2D numpy ndarray")

    idxs = np.unravel_index(np.argmax(X, axis=None), X.shape)

    i = idxs[0]
    j = idxs[1]

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    Parameters
    ----------
    n_terms : int
        number of factor in the wallis product

    Returns
    -------
    pi : float
        Wallis product pi approximation

    Raises
    ------
    ValueError
        If the input is not a positive int
    """
    if not isinstance(n_terms, int):
        raise ValueError("n_terms should be int")
    if not n_terms >= 0:
        raise ValueError("n_terms should be positive")

    id = 1 + np.arange(n_terms, dtype=np.float64)

    wallis_factors = (2 * id)**2 / ((2 * id - 1) * (2 * id + 1))

    return wallis_factors.prod() * 2
