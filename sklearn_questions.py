# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets



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
        self.X_ = X
        self.y_ = y
        # XXX fix
        return self

    def predict(self, X):
        """Write docstring
        """
        if X is None:
            raise ValueError

        check_is_fitted(self)
        X = check_array(X)
        arr_type = self.classes_.dtype
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0], dtype = arr_type)
        # XXX fix
        
        for index, x in enumerate(X):
            distances = np.linalg.norm(self.X_ - x, axis=1)
            y_pred[index] = self.y_[np.argmin(distances)]
        return y_pred

    def score(self, X, y):
        """Write docstring
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
