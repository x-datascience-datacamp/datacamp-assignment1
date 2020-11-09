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
    if type(X) is not np.ndarray:
        raise ValueError("the input is not a numpy")
    if (len(X.shape) != 2):
        raise ValueError("the shape is not 2D")
    R_max = []
    C_max = []
    for k in range(X.shape[0]):
        R_max.append(np.max(X[k][:]))
    X = X.T
    for k in range(X.shape[0]):
        C_max.append(np.max(X[k][:]))
    i = R_max.index(np.max(R_max))
    j = C_max.index(np.max(C_max))
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.
    if n_terms < 1:
        return 2.
    pi = 0.0
    for i in range(1, n_terms+1):
        print(i)
        x = 4 * (i ** 2)
        y = x - 1
        z = float(x) / float(y)
        if (i == 1):
            pi = z
        else:
            pi *= z
    pi *= 2
    return pi
