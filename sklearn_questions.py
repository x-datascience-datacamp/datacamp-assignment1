# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.neighbors import DistanceMetric


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Classifier implementing the 1-nearest neighbors vote."""
    
    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values.

        Parameters
        ----------
        X : array-like
            Training data, shape [n_samples, n_features]
        y : array-like
            Target values, shape [n_samples]

        Returns
        -------
        self : OneNearestNeighbor
            Fitted model.

        Raises
        -------
        ValueError
            If the input parameter y contains more than 50 classes.
        """
        X, y = check_X_y(X, y)
        X = check_array(X)
        self.classes_ = np.unique(y)

        if len(self.classes_) > 50:
            raise ValueError("Unknown label type: you have too many classes")

        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like
            Test samples.

        Returns
        -------
        y_pred : ndarray
            Class labels for each data sample.
        """
        check_is_fitted(self)
        X = check_array(X)

        dist = DistanceMetric.get_metric('minkowski')
        closest_neighbor = np.argmin(dist.pairwise(X, self.X_), axis=1)
        y_pred = self.y_[closest_neighbor]

        return y_pred

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like
            Test samples.
        y : array-like
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
