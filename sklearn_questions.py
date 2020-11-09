"""This is sklearn_questions."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Return the oneNearestNeighbor.

    Parameters:
        BaseEstimator: description.
        ClassifierMixin: description.

    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fitting the model.

        Parameters:
            X: Training Examples.
            y: Training labels.

        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """Predicting the Test samples.

        Parameters:
            self: Training Examples.
            X: Testing labels.

        """
        check_is_fitted(self)
        X = check_array(X)
        indices_1 = [np.sum(np.power(self.X_-t, 2), axis=1) for t in X]
        indices = np.argmin(indices_1, axis=1)
        return self.y_[indices]

    def score(self, X, y):
        """Get the score.

        Parameters:
            X: Testing Examples.
            y: Testing labels.

        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
