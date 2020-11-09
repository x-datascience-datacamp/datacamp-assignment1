# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Class constructor."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fits the data to the model.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features).
        y : ndarray of shape (n_samples).

        Returns
        -------
        self.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_ = X
        self.Y_ = y
        # XXX fix
        return self

    def predict(self, X):
        """Predicts the class of a given X point.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features).

        Returns
        -------
        y_pred: ndarray of shape(n_samples).
        """
        check_is_fitted(self)
        X = check_array(X)
        distances = np.zeros((len(X), len(self.X_)))
        for i in range(len(X)):
            for j in range(len(self.X_)):
                distances[i, j] = (np.linalg.norm(self.X_[j]-X[i], 2))
        closest = np.argmin(distances, axis=1)
        y_pred = np.full(shape=len(X), fill_value=self.Y_[closest])
        # XXX fix
        return y_pred

    def score(self, X, y):
        """Calculates the prediction score of the estimator.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features).
        y : ndarray of shape(n_samples).

        Returns
        -------
        Percentage of correct predictions.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
