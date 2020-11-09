# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets
from scipy.linalg import norm
import scipy


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Classifier implementing the 1-nearest neighbors vote

    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        Class labels known to the classifier

    https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """

        Parameters
        ----------
        X : Training data points
        y : Training labels

        Returns
        -------
        self (the object itself)
        """

        X, y = check_X_y(X, y, accept_sparse=True)
        check_classification_targets(y)

        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """

        Parameters
        ----------
        X : Data points to be labelled

        Returns
        -------
        y_pred : Label prediction

        """
        check_is_fitted(self)
        X = check_array(X)
        # y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        row_col = np.unravel_index(np.arange(len(self.X_) * len(X)), (len(self.X_), len(X)))

        distances = np.array([norm(self.X_[i] - X[j])
                              for i, j in zip(row_col[0], row_col[1])]).reshape(len(self.X_), len(X))
        indices = np.argmin(distances, axis=0)
        try:
            y_pred = self.y_[indices]
        except:
            y_pred = None

        return y_pred

    def score(self, X, y):
        """

        Parameters
        ----------
        X
        y

        Returns
        -------

        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
