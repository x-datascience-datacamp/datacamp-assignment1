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

    if X is None or type(X) is not np.ndarray:
        raise ValueError(
            "The input array X is None or not a numpy array.")

    if len(X.shape) != 2:
        raise ValueError(
            "The shape is not 2D.")

    s = X.shape[0]
    result = np.argmax(X)
    i = result // s
    j = result % s

    return i, j

def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """

    x = 2
    for n in range(1, n_terms+1):
        x *= ((4*n**2) / (4*n**2 - 1))

    return x

