# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from scipy.spatial import distance_matrix
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):

    """Classifier implementing the 1-nearest neighbor vote

    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        Class label meet in X_train
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the model using X as training data and y as class labels

        Parameters
        ----------
        X : array_like
            Training data, shape (n_samples, n_features)
        y : array-like
            class labels of shape = (n_samples,)

        Returns
        -------
        self : fitted model
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        check_classification_targets(self.classes_)
        self.obs_ = X
        self.labels_ = y
        return self

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_queries,)
            Class labels for each data sample.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        matrix = distance_matrix(X, self.obs_)
        min_dist = np.argmin(matrix, axis=1)
        y_pred = self.labels_[min_dist]
        return y_pred

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Test Samples
        y : ndarray of shape (n_queries,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
