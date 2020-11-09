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
    if X is None or not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise ValueError
    i_arr, j_arr = np.where(X == np.amax(X))
    # get first occurence
    i = i_arr[0]
    j = j_arr[0]
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.
    
    # catch edge cases
    if n_terms < 0:
        raise ValueError
    elif n_terms == 0:
        return 2
    # create range array
    arr = np.arange(1, n_terms+1)
    # calculate wallis formula
    return 2*(4*arr*arr/(4*arr*arr - 1)).cumprod()[-1]
