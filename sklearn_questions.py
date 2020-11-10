# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Write docstring
    """
    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Write docstring
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = np.unique(y)
        self.X = X
        self.y = y
        return self

    def predict(self, X):
        """Write docstring
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        # XXX fix
        y_pred = self.y[np.argmin(euclidean_distances(X, self.X), axis=1)]
        return y_pred

    def score(self, X, y):
        """Write docstring
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
