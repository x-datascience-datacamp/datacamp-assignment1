# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import type_of_target


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """A classifier implementing the 1-nearest neighbor for the euclidean distance.

    Attributes
    --
    classes_: array of shape (n_classes,).
        classes known to the classifier.

    X_: array or array-like.
        Points known to the classifier, the neighbors.

    y_: array or array-like
        Classes of the points know to the classifier.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Initialise the classes and the neigbors from the given data.

        Parameters
        --
        X: array or array-like of shape (n_neigbors, n_features)
            Coordinates for each of the neigbors

        y: array-like of shape (n_neigbors,)
            Classes for each of the neigbors
        """
        X, y = check_X_y(X, y)
        if type_of_target(y) not in ['binary', 'multiclass', 'unknown']:
            raise ValueError(f"Unknown label type: {y}")
        self.classes_ = np.unique(y)

        self.X_ = X.copy()
        self.y_ = y.copy()
        return self

    def predict(self, X):
        """Predict the class label for the given data.

        Parameters
        --
        X: array-like of shape (n_queries, n_features)

        Returns
        --
        y_pred: ndarray of shape (n_queries,) with values in classes_
           The predicted classes of the given data
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])

        def compd(q): return ((self.X_ - q)**2).sum(axis=1)
        distances = np.apply_along_axis(compd, 1, X)
        y_pred = self.y_[distances.argmin(axis=1)]

        return y_pred

    def score(self, X, y):
        """Compute mean accuracy for the given data.

        Parameters
        --
        X: array-like of shape (n_queries, n_features)
            Input data on which to perform prediction

        y: array-like of shape (n_queries,)
            True classes for the input `X` to compare with the prediction.

        Returns
        --
        acc: scalar value between 0 and 1
            the mean accuracy between the prediction and the truth `y`.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
