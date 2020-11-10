# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets

class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Classification algorithm that is non parametric
    """
    def __init__(self):

        pass

    def fit(self, X, y):
        """Fitting the model
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_= X
        self.y_ = y
        return self

    def predict(self, X):
        """Predict the classes of the vector of X

        Parameters
        ----------
        - self
        - X : array
            data to classify

        Returns
        -------
        y_pred : array
            the classes predict for each data X
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0],dtype= self.classes_.dtype)

        for j in range(X.shape[0]):
            distance_min = np.linalg.norm(X[j, :]-self.X_[0, :])
            index_min = 0
            for i in range(1,self.X_.shape[0]):
                if np.linalg.norm(X[j,:]-self.X_[i, :]) < distance_min:
                    index_min = i
                    distance_min = np.linalg.norm(X[j, :]-self.X_[i, :])
            y_pred[j] = self.y_[index_min]
        return y_pred

    def score(self, X, y):
        """The score of classification of X.

        Parameters
        ----------
        - self
        - X : array
            input data
        - Y : array
            clases corresponding on each input X

        Returns
        -------
        the mean off the error of missclassification
        """

        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
