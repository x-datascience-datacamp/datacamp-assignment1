# noqa: D100
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array

"""Regression classes threshold"""
reg_the = 30


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """1-Nearest-Neighbor implementation using sklearn API."""
    
    def __init__(self):  # noqa: D107
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model using X as input data and y labels.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            training data of shape
        y : ndarray of shap (n_samples,)
            target values of shape
        Returns
        ----------
        self : OneNearestNeighbor()
               the current instance of the classifier
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) > reg_the:
            raise ValueError(
                "Unknown label type: Classfifcation only allows a max of"
                + f" {reg_the}"
                )
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = len(X[0])
        check_classification_targets(y)
        return self

    def predict(self, X):
        """Predict the labels for the input X data.
        
        Parameters
        ----------
        X : ndarray of shape (n_test_samples, n_features)
            test data to predict on
        Returns
        ----------
        y : ndarray of shape (n_test_samples,)
            Class labels for each test data sample
        """
        check_is_fitted(self)
        X = check_array(X)
        if X[0].shape[0] != self.X_[0].shape[0]:
            raise ValueError("Number of feature in predict \
                             and in fit are different")
        D = cdist(X, self.X_)
        y_pred = np.asarray(list(map(lambda X: self.y_[np.argmin(X)], D)))
        return y_pred

    def score(self, X, y):
        """Compute the Mean Square Error for a given data and predicted labels.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data array.
        y : ndarray of shape (n_samples,)
            the true labels for X
        Returns
        -------
        score: float
               MSE value.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
