# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.metrics import pairwise_distances

class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """ One Nearest Neighbor class. """
    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor model on the data.

        Parameters
        ----------
        X : array of training data
        y : array of target values

        Returns
        ----------
        self : OneNearestNeighbor
               the current instance of the model

        Raises
        -------
        ValueError
            If the input parameter y contains more than 50 classes.
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) > 40:
            raise ValueError("Unknown label type: Regression Task")
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """Predict the labels for the input X data.

        Parameters
        ----------
        X : array of shape (n_test_samples, n_features)
            test data to predict on

        Returns
        ----------
        y : array of shape (n_test_samples,)
            Class labels for each test data sample
        """

        check_is_fitted(self)
        X = check_array(X)

        dist = pairwise_distances(X, self.X_)
        index = np.argmin(dist, axis=1)
        y_pred = self.y_[index]
        return y_pred

    def score(self, X, y):
        """Compute the mean accuracy for a given data and base prediction.

        Parameters
        ----------
        X : array of the input data
        y : array of the true labels for X

        Returns
        -------
        score: float
               mean accuracy
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
