# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.metrics.pairwise import pairwise_distances


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor.
    """
    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """fitting function.
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) > 30:
            raise ValueError("Unknown label type: Regression task")
        self.X_ = X
        self.y_ = y
        # XXX fix
        return self

    def predict(self, X):
        """predit function.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        all_distances = pairwise_distances(X, self.X_)
        closest_dist = np.argmin(all_distances, axis=1)
        y_pred = self.y_[closest_dist]
        return y_pred

    def score(self, X, y):
        """calculating scores.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
