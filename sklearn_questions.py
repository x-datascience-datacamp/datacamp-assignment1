# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from scipy.spatial import distance_matrix
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Creates a one nearest neighbour.
    """
    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fit the one NN 

        Parameters
        ----------
        X, y : training set and training labels 

        Returns
        -------
        self : the ONE NN 
        """
        
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        """Classifies X

        Parameters
        ----------
        X : training set 

        Returns
        -------
        the predicted class of X 
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        k_near = [np.argmin(np.linalg.norm(self.X_train_-x, axis=1)) for x in X]
        y_pred = np.array([self.y_train_[n] for n in k_near])
        return y_pred

    def score(self, X, y):
        """calculate the error of prediction 

        Parameters
        ----------
        X : array that we class
        y : the labels that is our target

        Returns
        -------
        the distance between the prediction and the label  

        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
