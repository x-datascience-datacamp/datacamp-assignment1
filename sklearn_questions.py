# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Class to find the nearest neighbor."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fits with the classifier.

         Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input array.
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) > 50:
            raise ValueError(
                "Unknown label type: Classfifcation max reached")
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """Predict output given an y.

        Parameters
        ----------
        X : ndarray of shape (n_features)
            The test input array.
        y : ndarray of shape (n_samples)
            The ground truth values.

        Returns
        -------
        Returns the predictions
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        neighbor_distances_and_indices = np.empty((len(X), len(self.X_)))

        k_nearest_distances_and_indices = np.empty((len(X)))

        for index, row_query in enumerate(X):
            for index_row, row in enumerate(self.X_):
                distance = np.linalg.norm(row - row_query)
                neighbor_distances_and_indices[index][index_row] = distance

        for index, row_query in enumerate(X):
            k_nearest_distances_and_indices[index] = np.argmin(
                neighbor_distances_and_indices[index])
        y_pred = np.array(
            [self.y_[int(i)] for i in k_nearest_distances_and_indices]
            )

        return y_pred

    def score(self, X, y):
        """Scoring metric.

         Parameters
         ----------
        X : ndarray of shape (n_samples, n_features)
            The input array.
        y : ndarray of shape (n_samples)
            The predictions

        Returns
        ----------
        Returns the score of the classifier as a float
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
