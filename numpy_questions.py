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
    if type(X) != np.ndarray:
        raise ValueError("the input is not a numpy")
    elif len(X.shape) != 2 or X.shape[1] < 1:
        raise ValueError("the shape is not 2D")
    else:
        for a in range(X.shape[0]):
            for b in range(X.shape[1]):
                if X[i][j] < X[a][b]:
                    i, j = a, b
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

     Parameters
    ----------
    n_terms : int
              The number of terms in the product.

    Returns
    -------
    pi_approx : float
        An approximation of pi.


    Raises
    ------
    ValueError
        If n_terms is not int.
    """
    if type(n_terms) is not int :
        raise ValueError("n_terms is not an integer")
    pi_approx = 1
    for n in range(1, n_terms+1):
        pi_approx *= (4*n*n)/((2*n-1)*(2*n+1))
    return 2*pi_approx
