# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Apply a 1-NN estimator to multiclass classification problem."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """X_train and y_train."""
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        check_classification_targets(self.classes_)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """Predict the class of the test data with euclidean distance min."""
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        y_pred = self.y_[np.argmin(euclidean_distances(X, self.X_), axis=1)]
        return y_pred

    def score(self, X, y):
        """Return the accuracy of our classification."""
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
