# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Write docstring.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Write docstring.
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        # XXX fix
        self.X = X
        self.y = y
        self.examples_ = [X, y]
        return self

    def predict(self, X):
        """Write docstring.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        # XXX fix
        for i, x in enumerate(X):
            closest = np.argmin(np.linalg.norm(self.examples_[0], 
            dtype=self.classes_.dtype))
            y_pred[i] = self.examples_[-1][closest]

        return y_pred

    def score(self, X, y):
        """Write docstring.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
