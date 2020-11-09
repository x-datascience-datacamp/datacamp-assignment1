# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """
    Implements k nearest neighbor classifier algorithm, with k=1.
    """
    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """
        Assigns inputs X and y to attributes "training" and "targets"

        Parameters
        ----------
        X : numpy.ndarray of shape 2 (n_samples, n_features) 
            The features
        y : numpy.ndarray of shape 2 (n_samples, ) 
            The targets

        Returns
        -------
        self : OneNearestNeighbor instance

        Raises
        ------
        ValueError
            If the inputs are not numpy array or
            If one of the shapes of inputs is not 2D or
            If the inputs don't have the same length or
            If length of the input is less than 2
        """
        X, y = check_X_y(X, y, ensure_min_samples=2)
        self.classes_ = np.unique(y)
        check_classification_targets(y)
        # XXX fix
        self.training_, self.targets_ = X, y
        return self

    def predict(self, X):
        """
        Returns the estimated target values from an input numpy.ndarray

        Parameters
        ----------
        X : numpy.ndarray of shape 2 (n_samples, n_features) 
            The features
        y : numpy.ndarray of shape 2 (n_samples, ) 
            The targets

        Returns
        -------
        y_pred : numpy.ndarray of shape (n_samples)
            The predicted classes

        Raises
        ------
        ValueError
            If length of input is less than 2

        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0], dtype=self.targets_.dtype)
        # XXX fix
        for i, x in enumerate(X):
            dist, ind = np.linalg.norm(x - self.training_[0], ord=None), 0
            for j, fts in enumerate(self.training_):
                new_dist = np.linalg.norm(x - fts, ord=None)
                if new_dist < dist:
                    dist, ind = new_dist, j
            y_pred[i] = self.targets_[ind]

        return np.array(y_pred)

    def score(self, X, y):
        """
        Computes accuracy of predicted values from X to real target values. Overrides ClassifierMixin.score().

        Parameters
        ----------
        X : numpy.ndarray of shape 2 (n_samples, n_features) 
            The features
        y : numpy.ndarray of shape 2 (n_samples, ) 
            The targets

        Returns
        -------
        numpy.float64
            The accuracy score

        Raises
        ------
        ValueError
            If length of input is less than 2
        """
        X, y = check_X_y(X, y, ensure_min_samples=2)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
