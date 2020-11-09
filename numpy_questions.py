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
    if type(X) is np.ndarray :
        raise Exception("the input is not a numpy array")
    if len(X.shape) != 2 :
        raise Exception("the shape is not 2D")
    i,j = np.argmax(X,axis=-1)
    return i, j



def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    Parameters
    ----------
    n_terms : the number of terms for the approximation of pi.

    Returns
    -------
    pi_approx : the approximate pi value with wallis product (n_terms times)
    
    Raises
    ------
    ValueError
        If n_terms < 0 
    """

    pi_approx = 1
    if n_terms < 0:
        raise ValueError("n_terms < 0")
        
    if n_terms == 0:
        return 2.0*pi_approx
        
    for i in range(1,n_terms):
        pi_approx *= ((4*i**2)/(4*i**2-1))
        
    return 2.0*pi_approx
