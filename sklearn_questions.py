# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Write docstring.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """To each point X return the class Y.

        Parameters :
        X : 2D array
        y : 1D array, represents the classes

        Returns :
        Nothing, it's a method that changes the attributes.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        # XXX fix
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """Return what is the class of a given X.

        Parameters :
            X : 2D array

        Returns :
            y : The class predicted.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        # XXX fix
        L = []
        for x in X:
            L.append(np.argmin(np.linalg.norm(x - self.X_, axis=1)))
        y_pred = np.array([self.y_[L[i]] for i in range(len(L))])
        return y_pred

    def score(self, X, y):
        """Return the accuracy of model.

        Parameters :
        X : a set of position, X_test
        y : a set of class, y_test

        Returns :
        Accuracy of model.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
