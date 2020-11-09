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
    if X is None:
        raise ValueError
    if isinstance(X,np.ndarray):
        i, j = np.unravel_index(np.argmax(X), X.shape)
    else : 
        raise ValueError
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.
    pi_demi = 1
    if n_terms!=0:
        for n in range(1, n_terms+1):
            pi_demi = pi_demi * 4*(n**2) / (4*(n**2)-1)

    return pi_demi*2
