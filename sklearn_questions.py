# noqa: D100
import numpy as np
from scipy.spatial import distance
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """
    Class to find the nearest neighbor

    Parameters
    ----------
    BaseEstimator : Base class for all estimators in scikit-learn.
    ClassifierMixin : Mixin class for all classifiers in scikit-learn.

    """
    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """
         Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input array.
        
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input array.
        y : ndarray of shape (n_samples)
            The ground truth values.

        Returns
        -------
        Returns the predictions
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        D = distance.squareform(distance.pdist(X))
        closest = np.argsort(D, axis=1)
        closest = closest[:, 1]
        print(closest)
        y_pred = closest
        return y_pred

    def score(self, X, y):
        """
         Parameters
         ----------
        X : ndarray of shape (n_samples, n_features)
            The input array.
        y : ndarray of shape (n_samples)
            The predictions

        Returns
        ----------
        Returns the score of the classifier
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
