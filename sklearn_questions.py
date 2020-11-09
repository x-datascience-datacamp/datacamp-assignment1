# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.metrics import euclidean_distances


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Create the classifier of One Nearest neighbor using sklearn."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """ Put the data into the empty model.

        Parameters
        ----------
        self : an empty classifier.
        X : ndarray of shape (n_samples, n_features)
            The input array.
        y : ndarray of shape (n_samples,)
            the traget array.
        Returns
        -------
        self : the classifier of the model
                after taking the training data into consideration.

        Raises
        ------
        ValueError : if the X is regression.

        """

        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        if len(self.classes_) > 50:
            raise ValueError(
                "Unknown label type: " +
                "Number of classe has passed the maximum number of classe")
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """
        Parameters
        ----------
        self : an empty classifier.
        X : ndarray of shape (n_samples, n_features)
            The input array.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
                with the prediction of each data sample.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[
                         0], dtype=self.classes_.dtype)
        for i, x in enumerate(X):
            closest = np.argmin(euclidean_distances([x], self.X_), axis=1)
            y_pred[i] = self.y_[closest][0]

        return y_pred

    def score(self, X, y):
        """
        Parameters
        ----------
        self : an empty classifier
        X : ndarray of shape (n_samples, n_features)
            The input testing array. (the new samples testing to predict)
        y : ndarray of shape (n_samples,)
            the traget testing array
            (the new target testing to compare the prediction)
        Returns
        -------
        score : accuracy of the prediction of the model.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
