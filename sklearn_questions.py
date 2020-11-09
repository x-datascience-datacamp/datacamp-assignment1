# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Write docstring
    test
    """
    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Write docstring
        test
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)      
        # XXX fix
        self.X_ = X.copy()
        self.y_ = y.copy()

        return self

    def predict(self, X):
        """Write docstring
        test
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        # XXX fix
        distances = np.apply_along_axis(((self.X_ - q)**2).sum(axis=1), 1, X)
        y_pred = self.y_[distances.argmin(axis=1)]
        return y_pred

    def score(self, X, y):
        """Write docstring
        test
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
