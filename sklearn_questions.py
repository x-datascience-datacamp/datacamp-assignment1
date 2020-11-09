# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import type_of_target


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Class of KNN classifier with K = 1."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit classifier data."""
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        # XXX fix
        if type_of_target(y) not in ['binary', 'multiclass', 'unknown']:
            raise ValueError(f"Unknown label type: {y}")

        self.train_data_ = X
        self.train_classes_ = y
        return self

    def predict(self, X):
        """Predict classes for X matrix.

        Returns:
        ----------
        y_pred : ndarray of shape (len(X))
            The classes for each data input array.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])

        def distance_function(x):
            return ((self.train_data_ - x)**2).sum(axis=1)
        distances = np.apply_along_axis(distance_function, 1, X)
        min_distances = np.argmin(distances, axis=1)
        y_pred = self.train_classes_[min_distances]
        return y_pred

    def score(self, X, y):
        """Calculate classifier score for the input data.

        Return the score value of the model.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
