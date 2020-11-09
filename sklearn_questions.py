# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Implement a 1-NN class with basic functions  (fit, predict, score)."""
    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Stores training data."""
        check_classification_targets(y)
        X, y = check_X_y(X, y, ensure_2d=True)
        self.classes_ = np.unique(y)
        self._X_train, self._y_train = X, y
        return self

    def predict(self, X):
        """Predict a label for each line of X."""
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X),
                         fill_value=self.classes_[0],
                         dtype=self.classes_.dtype)

        def compute_distance_array(X, x):
            """Compute the array of distances between each
            coordinate of X and x."""

            distances = [0]*len(X)
            for i in range(len(X)):
                distances[i] = np.linalg.norm(X[i] - x)
            return np.array(distances)

        for i in range(len(X)):
            distances = compute_distance_array(self._X_train, X[i])
            j = np.argmin(distances)
            y_pred[i] = self._y_train[j]
        return y_pred

    def score(self, X, y):
        """Return the accuracy on a test sample."""
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
