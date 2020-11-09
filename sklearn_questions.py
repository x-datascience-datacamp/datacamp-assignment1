# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Write docstring.

    Arguments
    ---------
    classes_.
    examples_.

    Methods
    -------
    fit.
    predict.
    score.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Train the One-Nearest-Neighbor classifier.

        Parameters
        ----------
        X : numpy ndarray
            The features
        y : numpy ndarray
            The target

        Returns
        -------
        self : the ONN
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.examples_ = [X, y]
        return self

    def predict(self, X):
        """Predict the label.

        Parameters
        ----------
        X : The test labels

        Returns
        -------
        y_pred : Vector of prediction
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(
            shape=len(X),
            fill_value=self.classes_[0],
            dtype=self.classes_.dtype
        )
        for i, x in enumerate(X):
            closest = np.argmin(np.linalg.norm(self.examples_[0]-x, axis=1))
            y_pred[i] = self.examples_[-1][closest]
        return y_pred

    def score(self, X, y):
        """Compute the score.

        Parameters
        ----------
        X : Vector of test features
        y : Vector of labels

        Returns
        -------
        np.mean(y_pred == y) : Mean accuracy
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
