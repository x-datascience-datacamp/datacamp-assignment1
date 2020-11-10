# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Classifier implementing the 1-nearest neighbors vote."""
    
    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array, shape (n_samples,)
            Target values.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test samples.


        Returns
        -------
        y : array, shape (n_samples,)
            Class labels for each data sample.
        """
        check_is_fitted(self)
        X = check_array(X)
        class_type = self.classes_.dtype
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0],
                         dtype=class_type)
        for i, x in enumerate(X):
            closest_index = np.argmin(np.linalg.norm(self.X_ - x, axis=1))
            y_pred[i] = self.y_[closest_index]

        return y_pred

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test samples.

        y : array-like, shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
                Mean accuracy of self.predict(X) wrt. y.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
