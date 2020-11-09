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
    if (type(X)!=type(np.array([0]))): 
        raise ValueError("the input is not a numpy error")
    if (X.ndim !=2 ):
        raise ValueError("the shape is not 2D")
    
    M = np.max(X)  
    i,j = np.where(np.isclose(X,M))
    i=i[0]
    j=j[0]
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    Parameters
    ----------
    n_terms : int
        The number of terms to compute.
        We take in consideration a float that is an integer.

    Returns
    -------
    pi : float
        The approximation of pi using the Wallis .

    Raises
    ------
    ValueError
        If the input is not a int or
        if the input is positive. 
    """
    pi = 2
    if (n_terms<0):
        raise ValueError("The input is negative")
    if(type(n_terms) != int) and(type(n_terms) != float) : 
        raise ValueError("The input is not an integer")
    if(n_terms!=int(n_terms)) :
        raise ValueError("The input is not an integer")
    n = int(n_terms)
    for i in range(1,n+1):
        pi *=   4 * i**2 /(4*i**2 - 1 )
    return pi
