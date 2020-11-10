# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Class for the One Nearest Neighbor estimator."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Store classes, training points and the associated labels."""
        check_classification_targets(y)
        X, y = check_X_y(X, y)

        self.classes_ = np.unique(y)

        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """Return the predicted class for every point in X."""
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X),
                         fill_value=self.classes_[0],
                         dtype=self.classes_.dtype)
        for j in range(len(X)):
            distances_to_points = np.array([np.linalg.norm(self.X_[i] - X[j])
                                           for i in range(len(self.X_))])
            nearest_neighbor_ind = np.argmin(distances_to_points)
            y_pred[j] = self.y_[nearest_neighbor_ind]
        return y_pred

    def score(self, X, y):
        """Compute the accuracy ratio for the prediction."""
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
