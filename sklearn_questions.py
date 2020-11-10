# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Class for one nearest neighbor model."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """
        Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.
        y : array-like of shape (n_samples,)
            True labels for X.
        
        Return
        -------
        score : object
            One nearest neighbor instance.
        """
        check_classification_targets(y)
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self._X = X
        self._y = y
        return self

    def predict(self, X):
        """
        Return the predictions of the model w.r.t the input.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.
        
        Return
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted labels.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0], dtype=self._y.dtype)
        for i in range(len(y_pred)):
            y_pred[i] = self._y[np.argmin(np.linalg.norm(self._X - X[i],
             axis=1))]
        return y_pred

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.

        Return
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
