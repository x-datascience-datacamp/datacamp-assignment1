# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.metrics import pairwise_distances


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """One Nearest Neighbor model."""

    def __init__(self):
        """Init function for model."""
        pass

    def fit(self, X, y):
        """Fit the model.

        Parameters
        ----------
        X : np array of training data.

        y : np array of training labels.

        Returns
        -------
        self : fitted model.

        Raises
        ------
        ValueError
            If the number of class is too large (over 50).

        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        if len(self.classes_) > 50:
            raise ValueError("Unknown label type: number of class too large.")

        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """Compute the prediction of an new observation.

        Parameters
        ----------
        X : np array of a new observation.

        Returns
        -------
        y_pred : int predicted label.
        """
        check_is_fitted(self)
        X = check_array(X)

        distances = pairwise_distances(X, self.X_)
        y_pred = self.y_[np.argmin(distances, axis=1)]

        return y_pred

    def score(self, X, y):
        """Compute the accuracy of the model on set of observations and true labels.

        Parameters
        ----------
        X : np array of set of observations.
        y : np array of set of labels.

        Returns
        -------
        accuracy : float.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
