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
    if not(isinstance(X,np.ndarray )):
        raise ValueError('you should enter an array')

    if len(X.shape)!=2:
        raise ValueError('you should enter a 2-D array')

    argmax=np.argmax(X)
    i,j=np.unravel_index(argmax, X.shape)

    return i,j




def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.
    prod=2
    for i in range(1,1+n_terms):
        a = (2 * i)/(2 * i - 1)
        b = (2 * i)/(2 * i + 1)
        total = a * b
        prod=prod*total
    return prod

