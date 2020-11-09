# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Class implementing the one nearest neighbor model
    """
    def __init__(self):  # noqa: D107
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model with the training data
        Parameters
        ----------
        X: np.ndarray
            Training features
        y: np.ndarray
            Training labels
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X: np.ndarray):
        """Use the one nearest neighbor to predict
        Parameters
        ----------
        X: np.ndarray
            Input features
        Return
        -------
        y_pred: np.ndarray
            The predicted labels
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = []  # with np full: three is stored as thr
        for i in range(len(X)):
            dists = np.linalg.norm(self.X_train_ - X[i], axis=1)
            idx_argmin = dists.argmin()
            y_pred.append(self.y_train_[idx_argmin])
        y_pred = np.asarray(y_pred)
        return y_pred

    def score(self, X: np.ndarray, y: np.ndarray):
        """Return the accuracy of the predictions
        Parameters
        ----------
        X: np.ndarray
            Input features
        y: np.ndarray
            Labels associated
        Return
        -------
        The accuracy (float) is computed
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
