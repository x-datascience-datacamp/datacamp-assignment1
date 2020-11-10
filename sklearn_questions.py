# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Predict the class of a sample after fitting on a dataframe."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Check X and y have same length and store them.

        Parameters
        ----------
        X : train ndarray of shape (n_samples, n_features)
        y: classes ndarray of size n_samples

        Returns
        -------
        self object

        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)

        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """Predict one neighbor for each test sample than get it's class.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            test data samples
        Returns
        -------
        y_pred : ndarray of shape n_samples
                predicted class for each sample

        Raises
        ------
        Exception
            If dont run fit before predict

        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0],
                         dtype=self.classes_.dtype)
        for i in range(len(X)):
            distance = np.array([np.linalg.norm(X[i] - x)
                                 for x in self.X_])  # Calculate the distance
            one_neighbor = np.argmin(distance)
            y_pred[i] = self.y_[one_neighbor]
        return y_pred

    def score(self, X, y):
        """Return the accuracy of model predictions over the dataset.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            test data samples
        y : ndarray of shape n_samples
            actual classes of each data sample

        Returns
        -------
        score : float value
                the accuracy of predictions over the dataset
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
