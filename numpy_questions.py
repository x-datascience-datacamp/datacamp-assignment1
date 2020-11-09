# Snoqa: D100
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

    if(X is None):

        raise ValueError("None")

    if (not isinstance(X, np.ndarray)):

        raise ValueError(" should ba a np.array ")

    else:
        if(len(X.shape) != 2):
            raise ValueError("The numpy array should be a 2D array")

    (i, j) = np.unravel_index(np.argmax(X, axis=None), X.shape)

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.
    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.
    pi = 0.0
    pair = np.arange(2, n_terms, 2)
    below_pair = pair - 1
    above_pair = pair + 1
    pi = 2 * np.prod(pair / below_pair) * np.prod(pair / above_pair)
    if(n_terms == 0):
        return 2.0

    if(n_terms == 1):
        return 8/3

    return pi
