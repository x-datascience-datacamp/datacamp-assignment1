# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import type_of_target
REGRESSION_CLASSES_THRESHOLD = 50


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """An implementation of a 1-Nearest Neighbor algorithm.

    Attributes
    --
    classes_: array of shape.
        classes known to the classifier.
    X_train_: array or array-like.
        Points known to the classifier, the neighbors.
    y_train_ : array or array-like.
        Classes of the points known to the classifier.
    """
    def __init__(self, params=None):  # noqa: D107
        self.params = params

    def fit(self, X, y):
        """Initialise the classes and the neigbors from the given data.

        Parameters
        --
        X: array of shape (n_neigbors, n_features)
            Coordinates for each of the neigbors
        y: array of shape (n_neigbors,)
            Classes for each of the neigbors
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if type_of_target(y) not in ['binary', 'multiclass', 'unknown']:
            raise ValueError(f"Unknown label type: {y}")
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        """Predict class labels.

        Parameters
        --
        X: array-like of shape (n_samples, n_features)
        Returns
        --
        y_pred: array of shape (n_samples,)
           The predicted labels
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        distances = euclidean_distances(X, self.X_train_)
        min_distances = np.argmin(distances, axis=1)
        y_pred = self.y_train_[min_distances]
        return y_pred

    def score(self, X, y):
        """Compute mean accuracy for the given data.

        Parameters
        --
        X: array-like of shape (n_samples, n_features)
            Input data to classify
        y: array-like of shape (n_samples,)
            Groundtruth labels of the input data
        Returns
        --
        acc: scalar
            the mean accuracy between predicted and the  groundtruth labels
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
