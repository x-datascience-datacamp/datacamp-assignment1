# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from math import inf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances

REGRESSION_CLASSES_THRESHOLD = 50


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """
        Class to build a 1-nearest neighbors classifier using scikit learn APIs.
    """

    def __init__(self):  # noqa: D107
        pass

    def get_nearest_neighbor_index(self, sample_datapoint):
        """
        Fit the OneNearestNeighbor model using X as input data and y as true labels

        Parameters
        ----------
        sample_datapoint : ndarray of shape (n_features,)
                           training data point to evaluate class on

        Returns
        ----------
        min_distance_index : int
                             the min distance index from all the observations
        """
        assert self.X_ is not None
        # get distances for current datapoint
        train_distances = euclidean_distances([sample_datapoint], self.X_)
        # removing the distance with the current point (which is close to 0)
        train_distances[0][0] = inf
        # find the index of the minimal distance (k=1)
        return np.argmin(train_distances)

    def fit(self, X, y):
        """
        Fit the OneNearestNeighbor model using X as input data and y as true labels

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

        if len(self.classes_) > REGRESSION_CLASSES_THRESHOLD:
            # Scikit learn imposes the unknows label type for REGRESSION_CLASSES_THRESHOLD
            raise ValueError(
                f"Unknown label type: Classfifcation only allows a max of {REGRESSION_CLASSES_THRESHOLD}"
            )

        self.X_ = X
        self.y_ = y
        self.y_fit_ = np.array(
            [
                self.y_[self.get_nearest_neighbor_index(sample_datapoint)] for i, sample_datapoint in enumerate(self.X_)
            ]
        )
        return self

    def predict(self, X):
        """
        Predict the labels for the input X data

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
        return np.array(
            [
                self.y_[self.get_nearest_neighbor_index(sample_datapoint)] for i, sample_datapoint in enumerate(X)
            ]
        )

    def score(self, X, y):
        """
        Computes the Mean Square Error for a given input data and base prediction

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
