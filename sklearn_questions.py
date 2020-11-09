# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """We write our own 1-Nearest neighbor classifier.

    The function will be comptible with the sklearn library. This classes inherites from the classe BaseEstimator and ClassifierMixin.
    """

    def __init__(self): # noqa: D107
        pass

    def fit(self, X, y):
        """Take the training data as arguments which can be one array in the case of unsupervised learning or two arrays in the case of supervised learning.
        
        Parameters
        ----------
        X : ndarray
            Data to train on.
        y : scalar
            response variable.
    
        Returns
        -------
        self.

        Raises
        ------
        X,y : shape are coherent.
        y : is y a category.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """Do the next classification on the data X.

            Parameters
        ----------
        X : ndarray
            Data to predict on.

        Returns
        -------
        y_pred : array
            prediction made on X.

        Raises
        ------
        X : Chake array.
        self : check if self is already fited.
        """
        check_is_fitted(self)
        X = check_array(X)
        fill = self.classes_[0]
        dtype = self.classes_.dtype
        y_pred = np.full(shape=len(X), fill_value=fill, dtype=dtype)
        for i, x in enumerate(X):
            closest = np.argmin(np.linalg.norm(self.X_-x, axis=1))
            y_pred[i] = self.y_[closest]
        return y_pred

    def score(self, X, y):
        """Compute performance of the method.

            Parameters
        ----------
        X : ndarray
            Data to predict on.

        Returns
        -------
        y : array
            real response value to compare prediction.

        Raises
        ------
        X,y : shape are coherent.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
