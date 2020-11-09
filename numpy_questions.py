# noqa: D100
import numpy as np

def max_index(x):
    """Return the index of the maximum in a numpy array.

    Parameters
    ----------
    x : ndarray
        The input array of shape (n_samples, n_features).

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
    if not isinstance(x,np.ndarray):
        raise ValueError("X is not an numpy array")
    if x.ndim != 2:
        raise ValueError("X shape is not 2D")

    max_indexes = np.argwhere(x==x.max())[0]
    return max_indexes[0],max_indexes[1]


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.
    eq = lambda n: 4*n**2 / (4*n**2 - 1)
    arr_n = np.array(range(1,n_terms+1))
    return np.prod(eq(arr_n))*2
