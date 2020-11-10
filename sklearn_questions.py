# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """One nearest neighbor classifier implementation

    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        Class labels known to the classifier

    X_ : array of shape (n_samples, [data_shape])
        Training data

    y_ : array of shape (n_samples, )
        Training data labels

    """
    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fits the model according to the training data

        Parameters
        ----------
        X : array of shape (n_samples, [data_shape])
            Training data
        y : array of shape (n_samples, )
            Training data labels

        Returns
        -------
        self
            Fitted one nearest neighbor classifier
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """Predicts the classes of a test data set

        Parameters
        ----------
        X : array of shape (n_test_samples, [data_shape])
            Testing data

        Returns
        -------
        y_pred : array of shape (n_test_samples, )
            Predicted labels
        """
        check_is_fitted(self)
        X = check_array(X)
        X = np.repeat(X[:, np.newaxis], np.shape(self.X_)[0], 1)
        X = np.argmin(np.linalg.norm(X - self.X_, axis=2), axis=1)
        def f(x): return self.y_[x]
        y_pred = f(X)
        return y_pred

    def score(self, X, y):
        """Returns the accuracy score of the model

        Parameters
        ----------
        X : array of shape (n_test_samples, [data_shape])
            Testing data
        y : array of shape (n_samples)
            Testing data labels

        Returns
        -------
        float
            Accuracy between 0 and 1
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
