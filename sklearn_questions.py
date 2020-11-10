# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.metrics import euclidean_distances


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):

    """Return the OneNearestNeighbor of a set of points."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):

        """Build of train model.
        Parameters:
        ----------
        X:ndarray represent the observed points X
        Y:ndarray represent the classes

        Return:
        ----------
        self: classifier
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        
        """Predict the class for a given obersed set of points X.
        Parameters
        ----------
        X : ndarray
              New  points which we want predict the class.
        Returns
        ----------
        y_pred: ndarray
                The predicted classes of the given X.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        # XXX fix
        dist = np.argmin(euclidean_distances(X, self.X_), axis=1)
        y_pred = self.y_[dist]
        return y_pred

    def score(self, X, y):
        """Display scores of classifier."""
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
