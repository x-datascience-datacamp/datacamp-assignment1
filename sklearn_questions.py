# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
import random

class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Write docstring
    """
    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Write docstring
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        # XXX fix
        select = random.randint(0, len(X) - 1)
        inX = X[select]
        diff_squared = (X - inX) ** 2
        euclidean_distances = np.sqrt(diff_squared.sum(axis=1))
        sorted_dist_indices = euclidean_distances.argsort()[:1]
        class_count = {}
        for i in sorted_dist_indices:
            vote_label = y[i]
            class_count[vote_label] = class_count.get(vote_label, 0) + 1
            
        # Descending sort the resulting dictionary by class counts
        sorted_class_count = sorted(class_count.items(),
                                   key=lambda kv: (kv[1], kv[0]),
                                   reverse=True)
        return sorted_class_count[0][0]

    def predict(self, X):
        """Write docstring
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        # XXX fix
        for sample in X:
            y_pred.append(self.fit(sample),y)
        # Loop through all samples, predict the class labels and store the results
        return y_pred

    def score(self, X, y):
        """Write docstring
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
