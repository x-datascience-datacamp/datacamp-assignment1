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
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise ValueError("X is not an array")

    i, j = np.unravel_index(np.argmax(X), X.shape)
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    Parameters
    ----------
    n_terms : int
            number of terms in the product

    Returns
    -------
    pi_approx : float
                An approximation of pi using the Wallis product
    """
    Wallis_product = 1
    for i in range(1, n_terms+1):
        Wallis_product *= (4*(i**2))/(4*(i**2)-1)
    return Wallis_product*2
