# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array
from sklearn.metrics import euclidean_distances



class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Write docstring
    """
    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Write docstring
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        # XXX fix
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """Write docstring
        """
        check_is_fitted(self, ['X_', 'y_', 'classes_'])
        X = check_array(X)

        distance = euclidean_distances(X, self.X_)
        indices = np.argmin(distance,axis=1)
        y_pred=self.y_[indices]
        #y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        # XXX fix
        return y_pred

    def score(self, X, y):
        """Write docstring
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
