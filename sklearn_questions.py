# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.metrics import pairwise_distances_argmin_min


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """
    Algorithm that creata an estimator based on a set of
    observations and ouputs
    It compares a the distance new observation
    """
    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fitting the One Nearest Neighbor Model
        Parameters
        X : np array of observations
        y : np array of labels
        Return the fitted model
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) > 50:
            raise ValueError("Unknown label type: ")
        self.observations_ = X
        self.targets_ = y  # XXX fix
        return self

    def predict(self, X):
        """
        Given an observation this methods gives the best
        estimator output based on the model
        The input is an observation and the output is the predicted y
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        min = pairwise_distances_argmin_min(X, self.observations_, axis=1)[0]
        y_pred = self.targets_[min]  # XXX fix
        return y_pred

    def score(self, X, y):
        """
        The method returns a float mesasuring the
        accuracy of the model on the training set
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
