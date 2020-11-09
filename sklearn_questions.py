# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Implements a one nearest neighbors classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Train the estimator on the train data.

        Parameters
        ----------
        X : The features
        y : The classes

        Returns
        -------
        The trained estimator
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """Predicts the classes for the test data.

        Parameters
        ----------
        X: The test data

        Returns
        -------
        y_pred: The predicted classes
        """
        check_is_fitted(self, ['X_', 'y_', 'classes_'])
        X = check_array(X)
        y_pred = np.full(
            shape=len(X), fill_value=self.classes_[0],
            dtype=self.classes_.dtype
                )
        for i in range(X.shape[0]):
            distances = []
            for j in range(self.X_.shape[0]):
                distances.append(np.sum(np.power(X[i]-self.X_[j], 2)))
            y_pred[i] = self.y_[distances.index(min(distances))]
        return y_pred

    def score(self, X, y):
        """Compute the score for the prediction.

        Parameters
        ----------
        X : The Features/data
        Y : The Classes
        Returns
        -------
        The score of the predictions
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
