# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.metrics.pairwise import euclidean_distances


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Creation of a OneNearestNeighbor."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Set the classifier which is fit using X(input) and Y(labels) .
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        Y : ndarray of shape (n_samples, n_features)
            Target data.
        Returns
        -------
        self : classifier.
        Raises
        ------
        ValueError
            If there are more than 50 classes (Regression case).
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        if len(self.classes_) > 50:
            raise ValueError("Unknown label type: ")

        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """Predict label of X.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            precdiction Input data.
        Returns
        -------
        y_pred : prediction from X.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0],
                         dtype=self.classes_.dtype)
        for i, ptn in enumerate(X):
            y_pred[i] = self.y_[np.argmin(
                euclidean_distances([ptn], self.X_), axis=1)][0]
        return y_pred

    def score(self, X, y):
        """Compute MSE between our prediction qnd real value.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
        Y : ndarray of shape (n_samples, n_features)
            True labels of X.
        Returns
        -------
        score : float
                MSE.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
