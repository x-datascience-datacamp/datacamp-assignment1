# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """ 
    One Nearest Neighbor class

    """
    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """for fitting the data
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """class prediction
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0],
                         dtype = self.classes_.dtype)
        for i in range(y_pred.shape[0]):
            euclidean_distances = np.linalg.norm(self.X_ - X[i,:], axis=1)
            index_closest = np.argmin(euclidean_distances)
            y_pred[i] = self.y_[index_closest]
        
        return y_pred

    def score(self, X, y):
        """Computes the score.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
