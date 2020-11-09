# noqa: D100
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array

from numpy_questions import max_index


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
        """Fit the model on the data.

        Returns
        -------
        self : OneNearestNeighbor
            The model
        """
        x, y = check_X_y(x, y)
        self.classes_ = np.unique(y)
        self.x_ = x
        self.y_ = y
        self.n_features_in_ = len(x[0])
        check_classification_targets(y)
        return self

    def predict(self, x):
        """Return model prediction on data.

        Parameters
        ----------
        x : np.ndarray
            The samples array

        Returns
        -------
        y_pred : np.ndarray
            The model prediction
        """
        check_is_fitted(self)
        x = check_array(x)
        if x[0].shape[0] != self.x_[0].shape[0]:
            raise ValueError("Number of feature in predict different than in fit")
        D = cdist(x, self.x_)
        y_pred = np.asarray(list(map(lambda x : self.y_[np.argmin(x)],D)))
        return y_pred

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
