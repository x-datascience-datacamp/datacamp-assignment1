# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """One-Nearest-Neighbor implementation.

    Methods
    -------
    fit():
       Fit the model on the data
    predict():
       Make prediction with the model
    score():
       Score the model
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, x: np.ndarray, y: np.ndarray):
        """Fit the model on the data."""
        x, y = check_X_y(x, y)
        self.classes_ = np.unique(y)
        self.x_ = x
        self.y_ = y
        self.n_features_in_ = len(self.classes_)
        check_classification_targets(y)
        return self

    def predict(self, x: np.ndarray):
        """Return model prediction on data."""
        check_is_fitted(self)
        x = check_array(x)
        if x[0].shape[0] != self.x_[0].shape[0]:
            raise ValueError("Number of feature in predict different than in fit")
        y_pred = []
        for i in range(len(x)):
            y_pred.append(self.y_[0])
            x_mean = np.mean(x[i])
            m = np.mean(self.x_[0])
            for k in range(len(self.x_)):
                x_fit_mean = np.mean(self.x_[k])
                if abs(x_mean - x_fit_mean) < abs(x_mean - m):
                    m = x_fit_mean
                    y_pred[i] = self.y_[k]
        return np.asarray(y_pred)

    def score(self, x, y):
        """Return model's score.
        Parameters
        ----------
        x : np.ndarray
            The samples array

        y : np.ndarray
            The features array

        Returns
        -------
        mean : int
            The mean score
        """
        x, y = check_X_y(x, y)
        y_pred = self.predict(x)
        return np.mean(y_pred == y)
