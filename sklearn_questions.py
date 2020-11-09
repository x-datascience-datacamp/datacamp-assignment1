# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor estimator class."""
    
    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Record the data used to fit the classifier."""
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_, self.y_ = X, y
        return self

    def predict(self, X):
        """Predict the classes associated with the elements of vector X."""
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0],
                         dtype=self.classes_.dtype)
        for k in range(len(X)):
            distances = np.linalg.norm(self.X_ - X[k, :], axis=1)
            index = np.argmin(distances)
            y_pred[k] = self.y_[index]

        return y_pred

    def score(self, X, y):
        """Compute the score of the algorithm on the data (X, y)."""
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
