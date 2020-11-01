import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Write docstring
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        """Write docstring
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) >= 100:
            raise ValueError('Unknown label type: ')
        # XXX fix
        self._X_train = X
        self._y_train = y
        return self

    def predict(self, X):
        """Write docstring
        """
        check_is_fitted(self)
        X = check_array(X)
        # XXX fix
        id_test = (
            (X[:, None] - self._X_train[None]) ** 2
        ).sum(axis=-1).argmin(axis=1)
        return self._y_train[id_test]

    def score(self, X, y):
        """Write docstring
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
