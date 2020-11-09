# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Write docstring."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Write docstring."""
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.trainingSet_ = [X, y]
        return self

    def predict(self, X):
        """Write docstring."""
        check_is_fitted(self, "trainingSet_")
        X = check_array(X)
        y_type = self.classes_.dtype
        lenX = len(X)
        y_pred = np.full(shape=lenX, fill_value=self.classes_[0], dtype=y_type)
        for i, X_i in enumerate(X):
            distances = np.linalg.norm(self.trainingSet_[0] - X_i, axis=1)
            nearest_id = np.argmin(distances)
            y_pred[i] = self.trainingSet_[1][nearest_id]
        return y_pred

    def score(self, X, y):
        """Write docstring."""
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
