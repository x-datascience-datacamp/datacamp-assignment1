# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """One Nearest Neighbor model."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """
        Train the model.

        X : ndarray of shape (n_samples, n_features)
            The training set of data
        y : ndarray of shape (n_samples, 1)
            The training set of labels

        Returns:
        self : the fitted model

        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        # XXX fix
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        """
        Perform prediction considering the model.

        self : model fitted

        X : ndarray
            The test set of data

        Returns:
        y_pred : np.array of predictions
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0],
                            dtype=self.classes_.dtype)
        # XXX fix
        for i, x in enumerate(X):
            closest_X = np.argmin(np.linalg.norm((self.X_train_ - x), axis=1))
            y_pred[i] = self.y_train_[closest_X]
        return y_pred

    def score(self, X, y):
        """
        Give the performance of the model.

        X : ndarray
            The test set of data
        y : ndarray
            The test set of labels

        Returns:
        score : the accuracy
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
