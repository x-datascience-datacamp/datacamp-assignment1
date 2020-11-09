# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Implement a 1-NN class with functions fit, predit and score."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Train the data."""
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.X_ = X.copy()
        self.y_ = y.copy()
        return self

    def predict(self, X):
        """Predicting the results for X using 1-NN method."""
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])

        def distance(vector):
            return ((self.X_ - vector)**2).sum(axis=1)
        index = np.argmin(distance(X))
        y_pred = self.y_[index]
        return y_pred

    def score(self, X, y):
        """Calculate the score of the prediction over the data set X,y."""
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
