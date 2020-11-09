# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_is_fitted


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Write docstring
    """
    
    """ parameters : """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Write docstring
        """
        """" X, y"""
        X, y = check_X_y(X, y)
        X = check_array(X, copy=True, ensure_2d=True)
        self.classes_ = np.unique(y)
        # XXX fix
        if len(self.classes_) > 50:
            raise ValueError(
                "Unknown label type: The are too many classes for classif")
        self.inp_ = X
        self.lab_ = y
        return self

    def predict(self, X):
        """Write docstring.
        """

        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        # XXX fix
        dist = pairwise_distances(X, self.inp_)
        indx = np.argmin(dist, axis=1)
        y_pred = self.lab_[indx]
        return y_pred

    def score(self, X, y):
        """Write docstring
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
