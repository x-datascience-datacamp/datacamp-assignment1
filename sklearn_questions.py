# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Algorithme des k plus proches voisins
    prend en entrée l estimateur et le classifieur
    renvoie les méthodes fit predict et score
    """
    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """fit X to y 
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(np.unique(y))>50:
            raise ValueError("Regression")
        # XXX fix
        return self

    def predict(self, X):
        """predict y with X
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        # XXX fix
        return y_pred

    def score(self, X, y):
        """Score y predicted with real y
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
