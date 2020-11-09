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

    if (isinstance(X, np.ndarray) is False) or X.ndim != 2:
        raise ValueError

    i, j = np.unravel_index(X.argmax(), X.shape)

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """
    if (isinstance(n_terms, int) is False) or n_terms < 0:
        raise ValueError

    if n_terms == 0:
        return(2)

    # Computing the wallis product
    elements = np.arange(1, n_terms+1)
    def function_1(x): return 4*(x**2) - 1
    def function_2(x): return 4*(x**2)

    vect_1 = np.apply_along_axis(function_1, axis=0, arr=elements)
    vect_2 = np.apply_along_axis(function_2, axis=0, arr=elements)
    product = np.prod(vect_2/vect_1)

    return 2*product
