# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from scipy.spatial import distance
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Our sklearn Estimator."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit part.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """Predict part.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0],
                         dtype=self.classes_.dtype)
        for idx, x in enumerate(X):
            d = np.array([distance.minkowski(x, k, 2) for k in self.X_])
            y_pred[idx] = self.y_[d.argmin()]
        return y_pred

    def score(self, X, y):
        """Score part.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
