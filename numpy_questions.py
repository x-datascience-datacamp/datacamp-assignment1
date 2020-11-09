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
    if type(X) is not np.ndarray :
        raise ValueError('the input is not a numpy')
  
    # TODO
    if (len(X.shape)!=2):
         raise ValueError('the shape is not 2D')
    rows_max=[]
    for r in range(X.shape[0]):
        rows_max.append(np.max(X[r][:]))
    i=np.where(X==np.max(rows_max))[0][0]
    j=np.where(X==np.max(rows_max))[1][0]
   # j=cols_max.index(np.max(cols_max))
    
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """
    pi_approx = 0.0   
    for i in range(1, n_terms+1):
        a = 4 * (i ** 2)
        b = a - 1
        z = float(a) / float(b)
        if (i == 1):
            pi_approx = z
        else:
            pi_approx *= z
    pi_approx *= 2
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.
    return pi_approx

A= np.array([[1, 4, 5], 
    [-5, 8, 9]])
i=np.where(A==1)[0][0]
print(max_index(A))