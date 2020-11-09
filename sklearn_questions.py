# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.metrics import pairwise_distances


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Write docstring
    """
    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Trainning the model.
        Parameters:
        X : Inputs of the model // Type = numpy array
        y : Training labels // Type = numpy array

        Output:
        self : Model that has been fitted
        """

        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        # XXX fix
        if len(self.classes_) > 50:
            raise ValueError("Unknown label type: Maximum number of classes reached")
        self.inputs_ = X
        self.labels_ = y
        return self


    def predict(self, X):
        """Write docstring
        """
        check_is_fitted(self)
        X = check_array(X)
        # y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        symmetric_matrix = pairwise_distances(X, self.inputs_)
        index_min = np.argmin(symmetric_matrix, axis=1)
        y_pred = self.labels_[index_min]
        return y_pred

    def score(self, X, y):
        """Write docstring
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
