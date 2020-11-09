# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.metrics import pairwise_distances_argmin_min


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """One Nearest Neighbor Model.

    ONN.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fitting the One Nearest Neighbor Model.

        Parameters
        ----------
        X : np array of observations
            Training set

        y : np array of labels
            Training labels

        Returns
        -------
        self : The fitted model
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) > 50:
            raise ValueError("Unknown label type: it's not classification")
        self.observations_ = X
        self.tagets_ = y
        return self

    def predict(self, X):
        """Predicts the label of an observation.

        Parameters
        ----------
        X : numpy array
            An observation we wish to classify

        Returns
        -------
        y_pre : int
                The predicted class of the observation X
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])

        min_index = pairwise_distances_argmin_min(X, self.observations_, axis=1)[0]
        y_pred = self.tagets_[min_index]

        return y_pred

    def score(self, X, y):
        """Compare ground truth with predicted class.

        Parameters
        ----------
        X : np array of new observations
        y : np array of ground truth labels

        Returns
        -------
        score : The pourcentage of correctly classified observations
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
