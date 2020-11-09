# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from scipy.spatial import distance_matrix
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """  One Nearest Neighbor Algorithm   
         Takes the estimator and the classifier as inputs 
         Returns fit, predict and score methods """
    
    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """ Fit the model to X and y """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        
        check_classification_targets(self.classes_)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """ Predicts y for a given X """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        # XXX fix
        y_pred = self.y_[np.argmin(distance_matrix(X, self.X_), axis=1)]
        return y_pred


    def score(self, X, y):
        """ Compare y_pred and real y to ive the score """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)




