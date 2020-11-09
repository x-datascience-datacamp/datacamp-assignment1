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
    if isinstance(X, (np.ndarray, np.generic))==False:
    	raise NameError() # "X is not an array\n"
    if X.ndim !=2:
    	raise NameError('Array shape is not 2D\n')
    i,j = np.unravel_index(np.argmax(X, axis=None), X.shape)
    return i, j


def wallis_product(n_terms):
	"""Implement the Wallis product to compute an approximation of pi.

	See:
	https://en.wikipedia.org/wiki/Wallis_product
	"""
	# XXX : The n_terms is an int that corresponds to the number of
	# terms in the product. For example 10000.
	"""Return an approximation of mathematical constant pi using Wallis product
	Parameters
	----------
	n_terms : Number of terms we use in the product. Must be a positive int

	Returns
	-------
	pi_approx : float
				our approximation of the number pi
	"""
	i = 1
	pi_approx = 1
	while i <=n_terms:
		pi_approx*=(4*i**2)/(4*i**2-1)
		i+=1
	return 2*pi_approx
