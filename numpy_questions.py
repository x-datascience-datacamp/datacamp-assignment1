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

    # To do
    if isinstance(X, np.ndarray) is False:
        raise ValueError('The input is not a numpy array')
    if len(X.shape) != 2:
        raise ValueError('The shape is not 2D')

    i = np.where(X == np.amax(X))[0][0]
    j = np.where(X == np.amax(X))[1][0]
    
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.
    Parameters
    ----------
    n_terms : int
            The number of terms in the product.

    Returns
    -------
    prod : float
        The Wallis product.

    Raises
    ------
    ValueError
        If the input is not an integer .
    """
    if isinstance(n_terms, int) is False:
        raise ValueError('The input is not an integer')
    if n_terms == 0:
        return 2
    if n_terms == 1:
        return 8 / 3
    prod = np.prod([(4 * n**2) / (4 * n**2 - 1) for n in range(1, n_terms+1)])
    return 2 * prod
