# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Return Nearest Neighbor.

    Parameters
    ----------
    BaseEstimator : Base class for all estimators in scikit-learn
    ClassifierMixin : Mixin class for all classifiers in scikit-learn.

    Returns
    -------

    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Train the OneNearestNeighbor Classifier

        """
        #Train the model : stocker X et y dans la classe
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        # XXX fix
        self.X_train , self.y_train = X , y
        return self

    def predict(self, X):
        """Write docstring
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        # Renvoyer Y_pred, X est un nouveau point, identifier le X_train le plus proche de ce point (calculer des distances)
        # XXX fix
        return y_pred

    def score(self, X, y):
        """Write docstring
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
