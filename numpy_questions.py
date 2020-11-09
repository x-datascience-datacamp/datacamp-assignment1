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

    if type(X) is not np.ndarray:
        raise ValueError('The input is not a numpy array')
    if X.ndim != 2:
        raise ValueError('The input array is note in 2D')
    i, j = np.where(X == X.max())
    if len(i) == 2:
        j = i[1]
        i = i[0]
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    Parameters
    ----------
    n_term : int
            The n_terms is an int that corresponds to the number of
            terms in the product. For example 10000.

    Returns
    -------
    pi_n : int
        The row index of the maximum.

    Raises
    ------
    ValueError
        If the input is not an integer
    """
    if n_terms - abs(n_terms) != 0:
        raise ValueError('The input is not an integer')

    pi_n = 1
    for i in range(1, n_terms+1):
        n_2 = 2*i
        pi_n *= n_2**2/((n_2-1)*(n_2+1))
    return 2*pi_n