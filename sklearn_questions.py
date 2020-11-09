# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Check if X and y have the correct shape.

    Parameters
    ----------
    X : numpy array of shape(n_samples, n_features)
        array of elements used for training.
    y : numpy array of shape(n_samples)
        array of classes for each sample of X.

    Returns
    -------
    Class OneNearestNeighbor.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Classify based on nearest neighbor.

        Parameters
        ----------
        BaseEstimator : class of estimators
        MixinClassifier : Mixin classifier
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        check_classification_targets(y)
        self.X_train_ = X
        self.Y_train_ = y
        return self

    def predict(self, X):
        """Classify X based on its nearest neigbor value.

        Parameters
        ----------
        X : numpy array
            array of elements to classify.

        Returns
        -------
        y_pred : numpy array
            the predicted class for each element of X.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])

        euclid_dist = euclidean_distances(X, self.X_train_)
        nearest = np.argmin(euclid_dist, axis=1)
        y_pred = self.Y_train_[nearest]
        return y_pred

    def score(self, X, y):
        """Calculate score.

        Parameters
        ----------
        X : numpy array of shape(n_samples, n_features)
            array of elements to classify.
        y : numpy array of shape(n_samples)

        Returns
        -------
        Accuracy of the predicted class.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
