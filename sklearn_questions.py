# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Class of the 1-NN estimator.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples_train, space_dim)
        The array of train points.

    y_ : ndarray of shape (n_samples_train)
        The array of train labels.

    classes_ : ndarray of shape (n_classes)
        The array of classes.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """To attribute labels to points and check that everything is ok.

        Inputs
        ------
        X : ndarray of shape (n_samples_train, space_dim)
            The array of train points.

        y : ndarray of shape (n_samples_train)
            The array of train labels.
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        # XXX fix
        check_classification_targets(y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """To return the prediction according to 1-NN estimator.

        Inputs
        ------
        X : ndarray of shape (n_samples_test, space_dim)
            The array of test points.

        Returns
        -------
        y_pred : ndarray of shape (n_samples_test)
            The array of predictions.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        # XXX fix
        y_pred = np.array([
            self.y_[np.argmin(np.linalg.norm(x - self.X_, axis=1))]for x in X])
        return y_pred

    def score(self, X, y):
        """To compute the score of the classification.

        Inputs
        ------
        X : ndarray of shape (n_samples_test, space_dim)
            The array of test points.

        y : ndarray of shape (n_samples_test)
            The array of test labels.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
