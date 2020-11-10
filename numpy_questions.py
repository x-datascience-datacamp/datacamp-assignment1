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

    print(X)
    print(len(np.shape(X)))

    if not isinstance(X, (np.ndarray, np.generic)) or len(np.shape(X)) != 2:
        raise ValueError

    i, j = np.where(X == np.max(X))
    print(i)
    print(j)

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.
    x = 1
    for i in range(1, n_terms+1):
        x = x * (4*i**2)/((4*i**2)-1)
    return x*2
