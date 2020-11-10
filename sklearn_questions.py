# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics.pairwise import euclidean_distances


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Model for getting the nearest neighbor with sklearn."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Train the model.

        Parameters
        ----------
        X : np array
            Input array.
        y : np array
            Labels array
        Returns
        -------
        self : model
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        if type_of_target(y) not in ['binary', 'multiclass', 'unknown']:
            raise ValueError(f"Unknown label type: {y}")
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """Predict outputs.

        Parameters
        ----------
        X : np array
            Input array
        Returns
        -------
        y_pred : predictions of the model
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0],
                         dtype=self.classes_.dtype)
        arg = np.argmin(euclidean_distances(X, self.X_), axis=1)
        y_pred = self.y_[arg]
        return y_pred

    def score(self, X, y):
        """Return the score of our model.

        Parameters
        ----------
        X : np array
            Input array.
        y : np array
            Labels array
        Returns
        -------
        score : float
                accuracy of our model.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
