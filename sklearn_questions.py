# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.metrics import pairwise_distances_argmin_min


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """One Nearest Neighbor classifier."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Return the model fitted on the training set.

        Parameters
        ----------
        X : 2D ndarray
            The samples of the training set.

        y : ndarray
            The labels of the training set.

        Returns
        -------
        self : Object
            The fitted model.

        Raises
        ------
        ValueError
            If the number of classes is too high.
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        # We store the training samples and labels
        if len(self.classes_) > 50:
            raise ValueError("Unknown label type: too much classes.")
        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """Predict the labels of a test set.

        Parameters
        ----------
        X : 2D ndarray
            Test set.

        Returns
        -------
        y_pred : 1D ndarray
            The predicted labels.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        # predict the class of the nearest neighbor for each test sample
        nn = pairwise_distances_argmin_min(X, self.X_, axis=1)[0]
        y_pred = self.y_[nn]

        return y_pred

    def score(self, X, y):
        """Return the score for prediction on the test set.

        Parameters
        ----------
        X: 2D ndarray
            Test set.

        y : 1D ndarray
            True labels of the test set.

        Returns
        -------
        score :
            Prediction score.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        score = np.mean(y_pred == y)
        return score
