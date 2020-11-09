# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.metrics import pairwise_distances


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """ An classifier which implements a 1-NN algorithm.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, 2)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fitting function for this classifier.

        Parameters
        ----------
        X : ndarray, shape (n_samples, 2)
            The training input samples.
        y : ndarray, shape (n_samples,)
            The training input labels.
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) > 50:
            raise ValueError(
                "Unknown label type: Regression task")
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """ Predict the class label for the given data

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            The predicted classes of the given data
        """
        check_is_fitted(self)
        X = check_array(X)
        distances = pairwise_distances(X, self.X_)
        idxs = np.argmin(distances, axis=1)
        y_pred = self.y_[idxs]
        return y_pred

    def score(self, X, y):
        """ Scoring function for the classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        y : ndarray, shape (n_samples,)
            The input labels.

        Returns
        -------
        _ : score of the predictions of the classifier on X
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
