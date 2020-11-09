# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):   
    """Predict the class y by using the closest neighbor of x.

    Parameters
    ----------
    BaseEstimator: class of estimators
    ClassifierMixin: Mixin classifier in scikit Learn

    Returns
    -------
    Class OneNearestNeighbor with the attributes:
        classes_ : the classes seen during fit
        X_train_ : train data
        Y_train_ : train classes
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Check that entries have correct shape.

        Parameters
        ----------
        X : numpy array
            train data
        y : numpy array
            train class
        Returns
        -------
        Class OneNearestNeighbor
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        check_classification_targets(y)

        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        """Predict classes for test data.

        Parameters
        ----------
        X: numpy array
            test data
        Returns
        -------
        y_pred: numpy array
            prediction for test data
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])

        distances = euclidean_distances(X, self.X_train_)
        closest_neighbor = np.argmin(distances, axis=1)
        y_pred = self.y_train_[closest_neighbor]
        return y_pred

    def score(self, X, y):
        """Return the error of prediction.

        Parameters
        ----------
        X: test data
        y: class of test data
        Return
        ----------
        error of prediction : float
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
