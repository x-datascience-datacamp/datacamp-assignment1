# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets
from scipy.spatial import distance_matrix


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Define a1-nearest neighbor classifier,"""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Train the model using the labelled data,

        X: observations of the training set (numpy array)
        y: labels of the training set (numpy array)
        ----------------------
        Parameters
        X: ndarray (nb_samples, nb_features)
        y : ndarray (nb_samples,)
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        """Predict the class label for new datapoints,

        ----------------------
        Parameters:
        X: Data Samples we want to get the label of
        ndarray (nb_data_points, nb_features)
        ----------------------
        Returns
        y:label (class) for each data sample - ndarray (nb_labels,)
        """
        check_is_fitted(self, ['X_train_', 'y_train_', 'classes_'])
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        dist_ = np.argmin(distance_matrix(X, self.X_train_), axis=1)
        y_pred = self.y_train_[dist_]
        return y_pred

    def score(self, X, y):
        """Compute the accuracy for a test sample,

        ----------------------
        Parameters:
        X : samples - ndarray (nb_samples, nb_features)
        y : true class for X - ndarray (nb_samples,)
        ----------------------
        Returns
        float corresponding to the mean accuracy of the class predicted for
        data X compared to value y
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
