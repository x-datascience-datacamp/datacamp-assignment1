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
    if X is None :
        raise ValueError("ValueError exception thrown")
    if not isinstance(X,np.ndarray):
        raise ValueError("Not a numpy array")
    if len(np.shape(X)) !=2:
        raise ValueError("ValueError exception thrown")
    l,c = np.shape(X)
    for line in range(0,l):
        for col in range(0,c):
            if X[line][col]>X[i][j]:
                i=line
                j=col

    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.
    answer=2
    if n_terms>0:
        answer *= 4/3
        for i in range(2,n_terms+1):
            answer*=(4*i*i)/(4*i*i-1)

    return answer
