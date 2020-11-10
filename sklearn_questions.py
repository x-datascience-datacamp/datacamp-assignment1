# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """This is the class of our estimator"""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """fit takes the training data and keep it in memory
        through attributes.
        ----
        Parameters :
        X : characteristic of the data train
        y : targets of the trainin set."""
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        # XXX fix
        self.X = X.copy()
        self.y = y.copy()
        return self

    def predict(self, X):
        """Gives the predict targets for a test set.
        ----
        Parameters :
        X : Characteristics of the data set.
        ----
        Return :
        y_pred : the predict set."""
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        # XXX fix
        y_pred = self.y[np.argmin(euclidean_distances(X, self.X), axis=1)]
        return y_pred

    def score(self, X, y):
        """Gives the score for a data set between the prediction
        made on this data set by the estimator and the real target
        ----
        Parameters :
        X : the characteristics of the data set
        y : the real tragets associated with the given data set X.
        ----
        Return :
        the score (i.e. the average of good predictions)"""
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
