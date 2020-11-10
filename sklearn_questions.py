"""This is the sklearn test."""

# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Implementation of One Nearest Neighbor.

    Args:
        BaseEstimator: generic estimator class.
        ClassifierMixin: Mixin class for all classifiers in scikit-learn.

    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Stocking the training set.

        Args:
            X: Training Examples.
            y: Training Labels.

        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        check_classification_targets(y)
        # Stock the training examples
        self.X_ = X
        self.Y_ = y
        return self

    def predict(self, X):
        """Find the nearest neighbors.

        Args:
            X: Testing set.
        Returns:
            y_pred: Prediction Class.

        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        ecl_distances = [np.sum(np.power(self.X_ - x, 2), axis=1) for x in X]
        neareset_neighbor = np.argmin(ecl_distances, axis=1)
        neareset_class = self.Y_[neareset_neighbor]

        y_pred = neareset_class
        return y_pred

    def score(self, X, y):
        """Accuracy of the model.

        Args:
            X: Validation set.
            y: Validation labels.

        Returns:
            score: Accuracy of the model.

        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        score = np.mean(y_pred == y)
        return score
