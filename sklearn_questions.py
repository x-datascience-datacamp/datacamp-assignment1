# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    "Class One Nearest Neighbor herites from BaseEstimato and ClassifierMixin."

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fits the model to the data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The observation matrix to be used in the training.
        y : ndarray of shape (n_samples)
            The classes vector to be used in the training.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_ = X
        self.Y_ = y
        return self

    def predict(self, X):
        """Predicts the class of data X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features).
            The observation matrix to be used in the prediction.

        Returns
        -------
        y_pred: ndarray of shape(n_samples)
                The predictions.
        """
        check_is_fitted(self)
        X = check_array(X)
        closest = []
        for i in range(len(X)):
            distances = np.linalg.norm(self.X_-X[i], axis=1)
            closest.append(np.argmin(distances))
        y_pred = np.full(shape=len(X), fill_value=self.Y_[closest])
        # XXX fix
        return y_pred

    def score(self, X, y):
        """Compute the score of the estimator.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features).
            The observation matrix to be used in the prediction.
        y : ndarray of shape(n_samples).
            The original classes.

        Returns
        -------
        The number of errors divided by the totla number.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
