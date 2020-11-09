# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """We write our own 1-Nearest neighbor classifier
    """
    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """takes the training data as arguments,
        which can be one array in the case of unsupervised learning,
        or two arrays in the case of supervised learning
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """does the next classification on the data X
        """
        check_is_fitted(self)
        X = check_array(X)
        fill = self.classes_[0]
        dtype = self.classes_.dtype
        y_pred = np.full(shape=len(X), fill_value=fill, dtype=dtype)
        for i, x in enumerate(X):
            closest = np.argmin(np.linalg.norm(self.X_-x, axis=1))
            y_pred[i] = self.y_[closest]
        return y_pred

    def score(self, X, y):
        """compute performance of the method
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
