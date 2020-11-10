# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """
    Classifier implementing the 1-nearest neighbors vote.

    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        Class labels known to the classifier

    trainingSet_: array of shape (n_samples,n_features,n_samples,n_ouputs).

    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """
        Fit the model using X as training data and y as target values.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The training data

        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The target values.

        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.trainingSet_ = [X, y]
        return self

    def predict(self, X):
        """
        Predict the class labels for the provided data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples.
        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Class labels for each data sample.

        """
        check_is_fitted(self, "trainingSet_")
        X = check_array(X)
        y_type = self.classes_.dtype
        lenX = len(X)
        y_pred = np.full(shape=lenX, fill_value=self.classes_[0], dtype=y_type)
        for i, X_i in enumerate(X):
            distances = np.linalg.norm(self.trainingSet_[0] - X_i, axis=1)
            nearest_id = np.argmin(distances)
            y_pred[i] = self.trainingSet_[1][nearest_id]
        return y_pred

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples.
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
