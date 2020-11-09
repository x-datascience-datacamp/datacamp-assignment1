# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import type_of_target


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """One-Nearest Neighbor Algorithm"""

    def __init__(self, params=None):  # noqa: D107
        self.params = params

    def fit(self, X, y):
        """Initialise the classes and the neigbors from the data.

        Args:
            X (numpy.ndarray): array of shape (n_neigbors, n_features)
            y (numpy.ndarray): array of shape (n_neigbors,)
        """

        if type_of_target(y) not in ['binary', 'multiclass', 'unknown']:
            raise ValueError

        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        """Predict class labels.

        Args:
            X (numpy.ndarray): array of shape (n_samples, n_features)

        Returns:
            y_pred: array of shape (n_samples,)
        """

        check_is_fitted(self)
        X = check_array(X)

        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        distances = euclidean_distances(X, self.X_train_)
        closest = np.argmin(distances, axis=1)
        y_pred = self.y_train_[closest]
        return y_pred

    def score(self, X, y):
        """Compute the mean accuracy for the data.

        Args:
            X (numpy.ndarray): array of shape (n_samples, n_features)
            y (numpy.ndarray): array of shape (n_samples,)

        Returns:
            accuracy (scalar): mean accuracy between predicted labels
            and the truth groundtruth labels
        """

        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
