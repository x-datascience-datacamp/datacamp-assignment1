# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Define class for the model One Nearest Neighbor."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the model."""
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """Return the predictions of the model."""
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X),
                         fill_value=self.classes_[0],
                         dtype=self.classes_.dtype)
        for i, x in enumerate(X):
            index = np.argmin(np.linalg.norm(self.X_ - x, axis=1))
            y_pred[i] = self.y_[index]
        return y_pred

    def score(self, X, y):
        """Return the score of the model."""
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
