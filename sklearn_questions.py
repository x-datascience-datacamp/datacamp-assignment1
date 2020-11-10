# noqa: D100

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """An implementation of 1-Nearest Neighbors classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit classifier to data.

        Here we do nothing more than store the values of X & Y.

        Parameters
        ----------
        X : np.ndarray(n,p)
            The features array of the train set
        y : np.ndarray(n,)
            The labels array of the test set

        Returns
        -------
        OneNearsetNeighbor()
            The instance of the classifier
        """

        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_ = X
        self.Y_ = y
        return self

    def predict(self, X):
        """Make the predictions of a given new dataset.

        We compute the euclidan distance to eachpoint,
         the label is the one of the closest point.

        Parameters
        ----------
        X : np.ndarray(m,p)
            The features array that we want to predict features.
        Returns
        -------
        np.ndarray(m,)
            The predicted features
        """
        check_is_fitted(self)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0],
                         dtype=self.classes_.dtype)
        for i in range(X.shape[0]):
            dist = np.linalg.norm(self.X_ - X[i, :], axis=1)
            y_pred[i] = self.Y_[np.argmin(dist, axis=0)]
        return y_pred

    def score(self, X, y):
        """Compute the score of a given validation set.

        Parameters
        ----------
        X : np.ndarray(n,p)
            The features array of the validation set
        y : np.ndarray(n,)
            The labels array of the validation set

        Returns
        -------
        np.float32
            The score of the classifier on the dataset
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
