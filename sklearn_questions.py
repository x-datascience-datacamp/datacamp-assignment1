# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array



class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Implemetation of nearest neighbour
    """
    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Copy the input data
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        # XXX fix
        self._X = X
        self._y = y
        return self

    def predict(self, X):
        """Find the NN and predict
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        # XXX fix
        for i in range(len(X)):
            NN_idx = np.argmin(np.linalg.norm(X[i, :] - self.X_, axis=1))
            y_pred[i] = self.y_[NN_idx]
        
        return y_pred

    def score(self, X, y):
        """Accurancy
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
