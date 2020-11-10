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
        raise ValueError('Expected X as  ndarray but got')
    else:
        if len(X.shape) != 2:
            raise ValueError('X should be a 2D numpy array')
        else:
            i, j = np.unravel_index(np.argmax(X, axis=None), X.shape)
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    Parameters
    ----------
    n_terms : positive integer
        the number of term in the product.

    Returns
    -------
    p : float
        TThe computed wallis product.

    """
    p = 2
    if n_terms == 0:
        return p
    else:
        for i in range(1, n_terms+1):
            p *= (4 * i ** 2) / (4 * i ** 2 - 1)
        return p
