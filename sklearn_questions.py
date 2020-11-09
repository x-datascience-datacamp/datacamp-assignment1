# noqa: D100
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor classifier.

    Simple implementation.
    """
    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fitting our model to train data.

        Save our parameters.
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) > 30:
            raise ValueError(
                "Unknown label type: Regression task")
        # XXX fix
        self.points_ = X
        self.labels_ = y
        return self

    def predict(self, X):
        """Predicting the label of a given test data.

        Return the list of labels.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        distances = pairwise_distances(X, self.points_)
        closest = np.argmin(distances, axis=1)
        y_pred = self.labels_[closest]
        # XXX fix
        return y_pred

    def score(self, X, y):
        """Computes the accuracy of the classifier.

        Return the score.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
