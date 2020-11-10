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
        If the input is not a numpy error orma
        if the shape is not 2D.
    """
    i = 0
    j = 0
    if ((isinstance(X,np.ndarray))==False or(len(X)==0 )or (len(X.shape) != 2)):
        raise ValueError(" The input must be an array of shape 2D")
    ind = np.argmax(X)
    i,j = np.unravel_index(ind,X.shape)
    
    # TODO

    return i,j

def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.
    if n_terms==0:
        return 2.0
    result=1
    if n_terms==1:
        return 8/3
    for count in range(1, n_terms+1):
        result*=((4*count*count)/((2*count-1)*(2*count+1)))
    return 2*result
