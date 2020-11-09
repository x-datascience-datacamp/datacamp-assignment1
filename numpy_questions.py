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
    try:
        assert isinstance(X, np.array(1))
        assert X.ndim == 2
    except AssertionError:
        raise ValueError
    max = np.unravel_index(np.argmax(X, axis=None), X.shape)
    i = max[0]
    j = max[1]
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    Parameters
    ----------
    n_terms : the number of terms for the calculation of pi.

    Returns
    -------
    pi : the approximation pi value calculated by the wallis product.

    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.
    pi = 1
    for i in range(1, n_terms+1):
        pi *= 4*(i**2)/(4*(i**2)-1)
    pi *= 2
    return pi
