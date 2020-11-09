# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):

    """Create the 1-N-N Classifier."""
    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Store trainning data."""
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        # XXX fix
        check_classification_targets(y)
        self.Xtrain_ = X
        self.ytrain_ = y
        return self

    def predict(self, X):
        """Predict the class of elements in the testing dataset X."""
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0],
                         dtype=self.classes_.dtype)
        # XXX fix
        for i in range(len(X)):
            L = []
            for k in range(len(self.Xtrain_)):
                L.append(np.linalg.norm(X[i]-self.Xtrain_[k]))
            y_pred[i] = self.ytrain_[np.argmin(L)]
        return y_pred

    def score(self, X, y):

        """Compare the calculated prediction of X y_pred
and the real value of the prediction y."""
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
