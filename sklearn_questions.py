# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
# Added Packages
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Algorithm of the One Nearest Neighbor."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Initialise the data and the neighbors.

        Parameters
        ----------
        X : ndarray of shape (n_neigbors, n_features)
        y : ndarray of shape (n_neigbors,)
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        check_classification_targets(y)
        self.X_train_ = X
        self.y_train_ = y

        return self

    def predict(self, X):
        """Predicts classes for the X data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        ----------
        y_pred : ndarray of shape (n_samples,)
        y_pred is the array of the predicted classes
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])

        dist = euclidean_distances(X, self.X_train_)
        closest_neighbor = np.argmin(dist, axis=1)
        y_pred = self.y_train_[closest_neighbor]
        return y_pred

    def score(self, X, y):
        """Return the mean value of the number of correct predictions.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_neigbors,)

        Returns
        ----------
        The value of the score of the prediction
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
