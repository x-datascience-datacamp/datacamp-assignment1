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
    
    if not isinstance(X, np.ndarray):
        raise Exception("not a numpy array")
       
    if  X.ndim!=2:
        raise Exception("not a 2D array")
         
    (i,j)= np.unravel_index(np.argmax(X, axis=None), X.shape)
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    Parameters
    ----------
    nterms : The n_terms is an int that corresponds to the number of
    terms in the product. For example 10000  .

    Returns
    -------
    Pi_wallis: approximation of pi (float)


    """
    Pi_wallis=1
    
    if not isinstance(n_terms, int):
       raise Exception("not an integer")
       
    if n_terms>0:
     for i in range(1,n_terms+1):
        Pi_wallis=Pi_wallis*(4*i**2)/(4*i**2-1)
    
       
    return 2*Pi_wallis
