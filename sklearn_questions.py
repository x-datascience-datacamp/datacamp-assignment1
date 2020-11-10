# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.metrics import pairwise_distances

from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Class to calculate the nearest neighbor."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """
        Fit the model using X as training data and y as labels.

        Parameters
        ----------------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples)
            True labels for X.

        Return
        ----------------
        self : OneNearestNeighbor
            The fitted model.
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        # XXX fix
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """
        Return the predictions of the model w.r.t the input.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training samples.

        Return
        ---------
        y_pred : ndarray of shape (n_samples)
            Predicted labels.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0],
                         dtype=self.classes_.dtype)
        # XXX fix

        y_pred = self.y_[np.argmin(pairwise_distances(X, self.X_), axis=1)]
        return y_pred

    def score(self, X, y):
        """
        Compute the accuracy of the model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training samples.
        y : ndarray of shape (n_samples)
            True labels for X.
        Returns
        -------
        score : float
            Accuracy.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
