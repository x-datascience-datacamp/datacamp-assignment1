import numpy as np
# noqa: D100


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
        If the input is not a numpy error orma
        if the shape is not 2D.
    """
    i = 0
    j = 0
    max_index_col = np.argmax(X, axis=0)
    max_index_row = np.argmax(X, axis=1)
    
    # TODO

    return max_index_col, max_index_row
max_index(np.array([[1,0,-3],[2,0.9,-1]]))


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.
    return 0.
