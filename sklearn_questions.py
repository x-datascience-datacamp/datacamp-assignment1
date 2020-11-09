# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import type_of_target


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor Classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor Classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input array.

        y : ndarray of shape (n_samples,)
            The labels.

        Returns
        -------
        self : object
            Returns self.

        """
        X, y = check_X_y(X, y)
        if type_of_target(y) not in ['binary', 'multiclass', 'unknown']:
            raise ValueError(f"Unknown label type: {y}")
        self.classes_ = np.unique(y)
        self.X_fit_ = X.copy()
        self.y_fit_ = y.copy()
        return self

    def predict(self, X):
        """Predicting function.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input array.

        Returns
        -------
        y : ndarray of shape (n_samples, 1)
            The predicted labels.

        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0],
                         dtype=self.classes_.dtype)
        for i, test_pt in enumerate(X):
            distances = np.array([np.linalg.norm(test_pt - fit_pt, 2)
                                  for fit_pt in self.X_fit_])
            nn = np.argmin(distances)
            y_pred[i] = self.y_fit_[nn]
        return y_pred

    def score(self, X, y):
        """Mean correct score of the classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input array.

        y : ndarray of shape (n_samples,)
            The labels.

        Returns
        -------
        self : object
            Returns self.

        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
