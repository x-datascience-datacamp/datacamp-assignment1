# noqa: D100

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import type_of_target


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """ 1-NearestNeightbor estimator."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """ Fit method the 1-NearestNeightbor
        Parameters
        ---------       
        X : ndarray of shape (n_samples, n_features)
            The input array.
        y : ndarray of shape (n_samples, 1)   
        Returns 
        --------
        self : OneNearestNeighbor
        """

        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        if type_of_target(y) not in ['binary', 'multiclass', 'unknown']:
            raise ValueError(f"Unknown label type: {y}")

        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        """Prediction of the class of X.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input array.
        Returns
        -------
        pred : ndarray of shape (len(X), 1)
                The output array with the predictions.
        """

        check_is_fitted(self)
        X = check_array(X)
        pred = np.full(shape=len(X), fill_value=self.classes_[0])
        dist = np.zeros((len(X)))

        for i in range(len(X)):
            dist[i] = np.argmin(np.linalg.norm(self.X_train_-X[i], axis=1))
        dist = dist.astype(int)

        pred = self.y_train_[dist]

        return pred

    def score(self, X, y):
        """Calculate the error rate of our predictor.
            Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input array on which we will predict.
        y : ndarray of shape (n_samples)
            The input array with correct classes
        Returns
        -------
        score : float 
                The score of the prediction.
        """
        X, y = check_X_y(X, y)
        pred = self.predict(X)
        return np.mean(pred == y)
