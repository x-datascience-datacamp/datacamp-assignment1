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
    if(type(X) is np.ndarray):
        if(len(X.shape)!=2):
            raise ValueError("Input is not a 2D np.ndarray")
    else:
        raise ValueError("Input is not a np.ndarray")

    max_index = np.argmax(X.flatten())
    
    i = max_index//X.shape[1]
    j = max_index%X.shape[1]

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.
    pi = 2.
    for i in range(1, n_terms):
        left_part = (2. * i)/(2. * i - 1.)
        right_part = (2. * i)/(2. * i + 1.)
        pi = pi * left_part * right_part
    
    return pi
