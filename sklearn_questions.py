# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.metrics import euclidean_distances


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Return Nearest Neighbor.

    Parameters
    ----------
    BaseEstimator : Base class for all estimators in scikit-learn
    ClassifierMixin : Mixin class for all classifiers in scikit-learn.

    Returns
    -------

    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Train the OneNearestNeighbor Classifier

        Parameters
        ----------
        X : Training data points
        y : Training labels

        Returns
        -------
        self(the object itself)
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        # XXX fix
        if len(self.classes_) > 0.8 * len(y) + 10:
            raise ValueError('y is regression')
        self.X_train, self.y_train = X, y
        return self

    def predict(self, X):
        """

        Parameters
        ----------
        X : Data points to be labelled

        Returns
        -------
        y_pred : Label predictions

        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        y_pred = self.y_[np.argmin(euclidean_distances(X, self.X_train), axis=1)]
        return y_pred

    def score(self, X, y):
        """Predict the class of the Test data.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
