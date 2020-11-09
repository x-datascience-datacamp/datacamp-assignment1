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
    RaisesP
    ------
    ValueError
        If the input is not a numpy error or
        if the shape is not 2D.
    """
    i = 0
    j = 0
    if type(X) is not np.ndarray:
        raise ValueError("X is not a numpy ndarray")
    if len(X.shape) != 2:
        raise ValueError("X is not a 2D array")
    i, j = np.unravel_index(np.argmax(X, axis = None), X.shape)
    return i, j





def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.
    See:
    https://en.wikipedia.org/wiki/Wallis_product
    XXX : write Parameters and Returns sections as above.
    """
    if type(n_terms) is not int:

         raise ValueError("n_terms is not an int")
    if n_terms < 0:
        raise ValueError("n_terms is negative")
    if n_terms == 0:
        result = 2.
    else :
       X = np.arange(1, n_terms+1, dtype=float)
       X = 4 * X**2 / (4 * X**2 - 1)
       pi = 2 * np.product(X)
    return pi
    
