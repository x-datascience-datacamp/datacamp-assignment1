# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Classifier implementing the one nearest neighbor."""
    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values
        Parameters
        ----------
        X : Training data. an  array with shape [n_samples, n_features].
        Y : Target values of shape = [n_samples].
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features],
            Test samples.
        Returns
        -------
        y : ndarray of shape [n_samples,]
            Class labels for each data sample.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0], dtype=self.classes_.dtype)
        for idx, X_test_row in enumerate(X):
            dist = []
            for idx_train, X_train_row in enumerate(self.X_train_):
                dist.append((idx_train, np.linalg.norm(X_train_row - X_test_row, ord=2)**.5))
            dist.sort(key=lambda tup: tup[1])
            y_pred[idx] = self.y_train_[dist[0][0]]

        return y_pred

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape [n_samples, n_features]
            Test samples.
        y : array-like of shape [n_samples,]
            True labels for X.
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
