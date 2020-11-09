# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import type_of_target


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Estimator to find the 1-nearest neighbor with SKlean."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Write docstring.

        Parameters
        ----------
        X : inputs
        y : labels

        Returns
        -------
        Fitted model
        """
        X, y = check_X_y(X, y)
        if type_of_target(y) not in ['binary', 'multiclass', 'unknown']:
            raise ValueError(f"Unknown label type: {y}")
        self.classes_ = np.unique(y)
        # XXX fix
        self.X_ = X.copy()
        self.y_ = y.copy()
        return self

    def predict(self, X):
        """Write docstring.

        Parameters
        ----------
        X : inputs
        Returns
        -------
        y_pred = 1-nearest neighbor
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        # XXX fix
        def compute_distance(v): return ((self.X_ - v)**2).sum(axis=1)
        distances = np.apply_along_axis(compute_distance, 1, X)
        y_pred = self.y_[distances.argmin(axis=1)]
        return y_pred

    def score(self, X, y):
        """Write docstring.

        Parameters
        ----------
        X : inputs
        y : labels

        Returns
        -------
        Mean of the good prediction
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
