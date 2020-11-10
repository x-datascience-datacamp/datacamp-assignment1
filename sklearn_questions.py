# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Return Nearest Neighbor."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Train the OneNearestNeighbor Classifier.

        Parameters
        ----------
        X : Training data points.
        y : Training labels.

        Returns
        -------
        self(the object itself)
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        check_classification_targets(self.classes_)
        self.X_, self.y_ = X, y
        return self

    def predict(self, X):
        """Predict the OneNearestNeighbor.

        Parameters
        ----------
        X : Data points to be labelled.

        Returns
        -------
        y_pred : Label predictions.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        dist = np.argmin(euclidean_distances(X, self.X_), axis=1)
        y_pred = self.y_[dist]
        return y_pred

    def score(self, X, y):
        """Return the accuracy."""
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
