# noqa: D100


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
    i = X.argmax(0)
    j = X.argmax(1)

<<<<<<< HEAD
    if type(X) != 'np.ndarray':
        raise TypeError('X is not an array')
    elif len(X.shape) != 2:
        raise TypeError('X is not a 2D vector')

    i = X.argmax(axis=0)
    j = X.argmax(axis=1)

=======
    if type(X) != 'numpy.ndarray':
        raise TypeError('X is not an array')

    if len(X.shape) != 2:
        raise TypeError('X is not a 2D vector')
    
>>>>>>> 7dcdf4f601e51aa732602f4bb7774ab53c130779
    return i, j


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    XXX : write Parameters and Returns sections as above.
    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.

    i = 1
    res = 1

    while i <= n_terms:
        res = res * (2 * (i) ^ 2) / ((2 * (i) ^ 2) - 1)

    return res
