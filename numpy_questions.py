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
    if isinstance(X, np.ndarray):
        if len(X.shape) == 2:
            i, j = np.argmax(np.max(X, axis=1)), np.argmax(np.max(X, axis=0))
        else:
            raise ValueError("Argument is not a 2D numpy array")   
    else:
       raise ValueError("Argument is not a numpy ndarray")
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    Parameters
    ----------
    n_terms : int
              The number of terms in the product. For example 10000.

    Returns
    -------
    pi_approx : int
                An approximation of pi.

    """
    n = np.arange(1, n_terms+1)
    pi_approx = 2 if n_terms == 0 else 2 * np.prod((2 * n / (2 * n - 1)) *
     (2 * n / (2 * n + 1)))

    return pi_approx
