# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array

class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Implemetation of nearest neighbour."""

    def __init__(self):  # noqa: D107
        """Initialize."""
        pass

    def fit(self, X, y):
        """Copy the input data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input array.
        y : ndarray of shape (n_samples, 1)
            The input labels.

        Returns
        -------
        self : iOneNearestNeighbor.
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        # XXX fix
        self._X = X
        self._y = y
        return self

    def predict(self, X):
        """Prediction class for X.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input array.

        Returns
        -------
        y_pred : ndarray of shape (n_samples, 1)
                 The arry of predicted class.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        # XXX fix
        for i in range(len(X)):
            NN_idx = np.argmin(np.linalg.norm(self._X - X[i], axis=1))
            y_pred[i] = self._y[NN_idx]
        return y_pred

    def score(self, X, y):
        """Accurancy.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input array.

        y : ndarray of shape (n_samples, 1)
            The input labels.

        Returns
        -------
        score : float
                 The score of the prediction.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
