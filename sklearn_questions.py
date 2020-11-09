# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Return the Nearest Neighbor of a set of points.

    Parameters
    ----------
    X : ndarray
        Known points.

    y : ndarray
        Known classes.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Saving the known point to initializate the estimator.
        --
        X : ndarray
            Coordinates for each of the neigbors
        y : ndarray
            Classes for each of the neigbors
        """
        X, y = check_X_y(X, y)
        y = check_classification_targets(y)
        self.classes_ = np.unique(y)
        # XXX fix
        self.X = X
        self.y = y
        return self

    def predict(self, X):
        """Predict the class for a given data X.

        Parameters
        ----------
        X : ndarray
            coordinates of the points who we want predict the class.

        Returns
        -------
        y_pred: ndarray
                The predicted classes of the given points.
        """

        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        # XXX fix
        for i, x in enumerate(X):
            dist = np.array([np.linalg.norm(d) for d in self.X-x])
            y_pred[i] = self.y[dist.argmin()]
        return y_pred

    def score(self, X, y):
        """Scores the classifier."""
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
