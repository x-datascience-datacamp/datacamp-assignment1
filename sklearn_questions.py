# %%
# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array

# %%


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Class to find the nearest neighbor

    Parameters
    ----------
    BaseEstimator : Base class for all estimators in scikit-learn.

    ClassifierMixin : Mixin class for all classifiers in scikit-learn.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Method which store X and y as attribute of the class 

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The features input array.

        y : ndarray of shape(n_samples,1)
            The Targets input array

        Returns
        -------
        self : OneNearestNeighbor(BaseEstimator, ClassifierMixin)
            The instance of the class
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        check_classification_targets(y)
        self.X_ = X
        self.Y_ = y
        return self

    def predict(self, X):
        """Method which return the nearest neighbor of X

        Parameters
        ----------
        X: ndarray of shape (1,n_features)
           The features input array

        Returns
        -------
        y_pred : int
                 The prediction
        """
        check_is_fitted(self)
        X = check_array(X)
        i = np.argmin(euclidean_distances(X, self.X_), axis=1)
        y_pred = self.Y_[i]
        return y_pred

    def score(self, X, y):
        """Method which return the score of the prediction

        Parameters
        ----------

        X: ndarray of shape (n_samples,n_features)
           The features input array

        y : ndarray of shape(n_samples,1)
            The Targets input array

        Returns
        -------
        y_pred : float
                 The score
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
