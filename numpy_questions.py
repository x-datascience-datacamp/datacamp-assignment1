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
    if (len(X.shape) != 2):
        raise ValueError("We need 2D Array!")

    i,j=np.argmax(X)
    return i, j

def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.

    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.
    wallis_pi=1
    for k in range(1,n_terms):
        wallis_pi*=4 * k ** 2 / (4 * k ** 2 - 1)
    return wallis_pi*2


print("Our estimation of Pi is: ", wallis_product(10000))
print("The absolute value of the difference is equal to:",abs(wallis_product(10000)-np.pi))