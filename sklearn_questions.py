# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """A suggestion of One Nearest Neighbor implementation."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Training phase : store observations and labels.

        Parameters
        ----------
        X : observations, ndarray of shape (n_samples, n_features)
        y : labels, ndarray of shape (n_samples,)
        Returns
        -------
        self : fitted model.

        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        # XXX fix
        self.observations_ = X
        self.labels_ = y
        return self

    def predict(self, X):
        """Prediction phase : classify new testing data.

        Parameters
        ----------
        X : new observations, ndarray of shape (n_samples, n_features)
        Returns
        -------
        y : predicted labels, ndarray of shape (n_samples,)

        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        # XXX fix

        for i in range(X.shape[0]):
            n = self.observations_.shape[0]
            dist = np.zeros(n)
            for j in range(n):
                dist[j] = np.sum(np.square(X[i] - self.observations_[j]))
            y_pred[i] = self.labels_[np.argmin(dist)]

        return y_pred

    def score(self, X, y):
        """Evaluate our classification.

        Parameters
        ----------
        X : observations, ndarray of shape (n_samples, n_features)
        y : true labels, ndarray of shape (n_samples,)
        Returns
        -------
        np.mean(y_pred == y) : float, percentage of well predicted labels.

        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
