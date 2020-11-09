# noqa: D100

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import type_of_target


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Creates a One NearestNeightbor simulator."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the 1-NN i.e. memorizes the X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input array.

        Returns
        -------
        self : OneNearestNeighbor

        """

        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        if type_of_target(y) not in ['binary', 'multiclass', 'unknown']:
            raise ValueError(f"Unknown label type: {y}")

        self.train_features_ = X
        self.respective_label_ = y
        return self

    def predict(self, X):
        """Predict the class of X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input array.

        Returns
        -------
        y_pred : ndarray of shape (len(X))
                The output array with predicted classes.

        """

        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        var = np.zeros((len(X)))

        for i in range(len(X)):
            var[i] = np.argmin(np.linalg.norm(self.train_features_-X[i],
                               axis=1))
        var = var.astype(int)

        y_pred = self.respective_label_[var]

        return y_pred

    def score(self, X, y):
        """Calculate the error rate of our predictor.

            Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input array on which we will predict.
        y : ndarray of shape (n_samples)
            The input array with correct classes

        Returns
        -------
        score : float
                The predictor score.

        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
