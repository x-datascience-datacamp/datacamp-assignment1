# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics import euclidean_distances


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Creates a one nearest neighbour."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit X with a one nearest neighbhor.

        Parameters
        ----------
        X : training set

        Returns
        -------
        The train set.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        """Predict the class of X.

        Parameters
        ----------
        X : set we want to classifies.

        Returns
        -------
        y_pred : the prediction of the classificattion

        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        closest = np.argmin(euclidean_distances(X, self.X_train_), axis=1)
        y_pred = self.y_train_[closest]
        return y_pred

    def score(self, X, y):
        """Give the distance between the prediction and the labels.
        
        Parameters
        ----------
        X : the set classified 
        y : the label

        Returns
         -------
        The distance.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
