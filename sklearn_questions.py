# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin): 
    """
    Classifier implementing the k-nearest neighbors vote.

    Attributes
    ----------
    classes_ : array
        Distinct labels used by the classifier

    X_ : array or matrix of shape (n_training_points, n_features)
        Data for train.

    y_ : array or matrix of shape (n_training_points,)
        Labels of training data points
    """
    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """
        Parameters
        ------
        X : array or matrix of shape (n_training_points, n_features)
            Data to train.
            
        y : array or matrix of shape (n_training_points,)
            Labels.
        
        Returns
        ------
        self : OneNearestNeighbor()
            A fitted One-nearest neighbor class
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
        ------
        X : array-like of shape (n_testing_points, n_features)
            Test data.

        Returns
        ------
        y_pred : array of shape (n_testing_points,) 
            Predicted labels for test data points.

        Raises
        ------
        ValueError
            If X is None
        """
        if X is None:
            raise ValueError

        check_is_fitted(self)
        X = check_array(X)
        arr_type = self.classes_.dtype
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0], 
                         dtype=arr_type)
        for index, x in enumerate(X):
            distances = np.linalg.norm(self.X_ - x, axis=1)
            y_pred[index] = self.y_[np.argmin(distances)]
        return y_pred

    def score(self, X, y):
        """Accuracy score of the classification
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
