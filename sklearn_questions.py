# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import euclidean_distances


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Use the nearest neighbor to do the classification."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Use the data to fit the model."""
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """Predict the cluster of X.""" 
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        distance = euclidean_distances(X, self.X_)
        y_pred = self.y_[np.argmin(distance, axis=1)]
        return y_pred

    def score(self, X, y):
        """Compute mean accuracy."""
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
