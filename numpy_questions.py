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

    if not isinstance(X, np.ndarray):
        raise ValueError("You did not enter an array.")
    if len(X.shape) != 2:
        raise ValueError("You did not enter an array with dim 2")
    (i, j) = np.unravel_index(X.argmax(), X.shape)

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """
    result = 1
    for k in range(1, n_terms + 1):
        result *= (4*k**2)/(4*k**2 - 1)
    return 2 * result
