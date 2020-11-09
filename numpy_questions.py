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

    if type(X)!= numpy.ndarray :
        raise ValueError("the input should be an ndarray ")
    if X.ndim !=2:
        raise ValueError('the shape of the input is not 2D')
    a= X.index(max(X))
    i=a[0]
    j=a[1]
    # TODO

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.
    a=1
    for i in range(n_terms):
        a=a*((2*n/(2*n-1))*(2*n/(2*n+1)))

    return a
