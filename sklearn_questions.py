# noqa: D100
import numpy as np
from numpy.core.fromnumeric import shape
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    def __init__(self):  # noqa: D107
        """Initialisation method of OneNearestNeighbor class.

        Parameters
        ----------
        None

        Returns
        -------
        An instance of OneNearestNeighbor class.

        Raises
        ------
        Nothing
        """
        pass

    def fit(self, X, y):
        """Fit the OneNearestNeighbor estimator

        Parameters
        ----------
        X : 2D numpy array containing the training points
        y : 1D numpy array containing the classes of the training points

        Returns
        -------
        Nothing

        Raises
        ------
        ValueError
            If the X input is not a numpy array or it's shape is not 2D.
            If the y input is not a numpy array or it's shape is not 1D.
            If the number of rows of X and y do not match.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_ = X
        self.Y_ = y
        return self

    def predict(self, X):
        """Predicts a class for the X input

        Parameters
        ----------
        X : 2D numpy array containing the training points

        Returns
        -------
        The class associated with the nearest training data point

        Raises
        ------
        ValueError
            If the X input is not a numpy array or it's shape is not 2D.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0],
                         dtype=self.classes_.dtype)
        for i in range(shape(X)[0]):
            closest = 0
            min_dist = np.linalg.norm(X[i, :] - self.X_[0, :])
            for j in range(1, shape(self.X_)[0]):
                if np.linalg.norm(X[i, :] - self.X_[j, :]) < min_dist:
                    closest = j
                    min_dist = np.linalg.norm(X[i, :] - self.X_[j, :])
            y_pred[i] = self.Y_[closest]
        return y_pred

    def score(self, X, y):
        """Predicts a class for the X input

        Parameters
        ----------
        None.

        Returns
        -------
        The score of the OneNearestNeighbor estimator which is the proportion
        of the rightly predicted classes among all predicted classes

        Raises
        ------
        ValueError
            If the X input is not a numpy array or it's shape is not 2D.
            If the y input is not a numpy array or it's shape is not 1D.
            If the number of rows of X and y do not match.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
