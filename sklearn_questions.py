# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.metrics import pairwise_distances


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """1-nearest neighbors classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit X to y.

        Parameters
        -------------------------
        X : training data - ndarray of shape ( n_samples, n_features )
        y : target value - ndarray of shape ( n_samples )
        Returns
        -------------------------
        self
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) > 50:
            raise ValueError("Unknown label type: Regression task")
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """Predict the class for a givin X.

        Parameters
        ----------------------
        X : array, shape (n_samples, n_features).

        Returns
        -----------------------
        y_pred ; perdicted class
        """
        check_is_fitted(self)
        X = check_array(X)
        Distances = pairwise_distances(X, self.X_)
        y_pred = self.y_[np.argmin(Distances, axis=1)]
        return y_pred

    def score(self, X, y):
        """Scoring function.

        Parameters
        -----------------------------
        x : array-like, shape (n_samples, n_features)
            The input samples

        y : ndarray, shape (n_samples,)

        Returns
        -----------------------------
        score
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
