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
        amax = np.argmax(X)
        i = amax // X.shape[1]
        j = amax % X.shape[1]
    except:
        raise ValueError("")

    return i, j

def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """

    X = np.arange(1, n_terms+1, dtype=float)
    X = 4 * X**2 / (4 * X**2 - 1)
    p = 2 * np.product(X)

    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.
    return p
