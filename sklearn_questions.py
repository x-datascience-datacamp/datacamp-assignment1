# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.metrics.pairwise import euclidean_distances


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Creating OneNearestNeighbor classifier.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the estimator.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data array.
        y : ndarray of shape (n_samples,)
            the true labels for X
        Returns
        -------
        self : OneNearestNeighbor() instance of the classifier
        """

        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) > 50:
            raise ValueError("Unknown label type: ")

        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """Predict the given data X based on self.X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data array.
        Returns
        -------
        y_pred : the predicted labels for X
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])

        distances = euclidean_distances(X, self.X_)
        distances = [np.argmin(distances[i])
                     for i in range(distances.shape[0])]
        y_pred = self.y_[distances]

        return y_pred

    def score(self, X, y):
        """Compute the Mean for a given data and base prediction.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data array.
        y : ndarray of shape (n_samples,)
            the true labels for X
        Returns
        -------
        score: float Mean score.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
