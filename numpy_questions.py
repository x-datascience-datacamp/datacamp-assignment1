# noqa: D100
import sys
import numpy as np


def max_index(X):
    """Return the index of the maximum in a numpy array.
!
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
 
    try: 
        max = np.argmax(X,axis=1)
        i = max[0]
        j = max[1]
    except : 
        if (type(X) is not np.array): 
            raise ValueError("Numpy error")
        if (type(X) is None) : 
            raise ValueError("None")
        if (len(X) != 2): 
            raise ValueError("Not 2D Shape")
  
  
  
    return i, j
    
    #raise ValueError ('bonjour')


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.


    
    return 0.
"""
a = np.random.randn(100, 100)
print(len(a))
i,j = max_index(a)
print(i)
print(j)
"""
#assert np.all(a[i, j] >= a)
