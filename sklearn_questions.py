# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """résoud un problème de classification, grâce à un algorythme des plus proches voisins
    """
    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Pretraitement des données
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        """Prédiction de la classe des données X
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        nearest = [np.argmin(np.linalg.norm(self.X_train_-x, axis=1)) for x in X]
        y_pred = np.array([self.y_train_[n] for n in nearest])
        return y_pred

    def score(self, X, y):
        """Donne l'erreur de la classification prédite sur les data X
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
