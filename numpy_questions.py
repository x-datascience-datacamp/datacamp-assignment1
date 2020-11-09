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
    ind = np.unravel_index(np.argmax(X), X.shape)
    i = ind[0]
    j = ind[1]
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.

    if not isinstance(n_terms, int):
        raise ValueError("n_terms should be an integer")
    if n_terms < 0:
        raise ValueError("n_terms should be at least 0")
    pi_approx = 1.0
    for n in range(1, n_terms+1):
        pi_approx = pi_approx * 4*n**2/(4*n**2-1)
    return 2*pi_approx
