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

    try:
        shape = X.shape
    except AttributeError:
        raise ValueError("The inupt should be numpy array-type")

    if len(shape) != 2:
        raise ValueError("The input should be 2D "
                         "(not " + str(len(shape)) + "D")

    i, j = np.unravel_index(X.argmax(), X.shape)

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.

    if n_terms == 0:
        return 2

    else:
        terms = np.int64(np.arange(n_terms) + 1)
        terms = 4*terms**2 / (4*terms**2 - 1)
        return 2 * np.prod(terms)
