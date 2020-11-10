# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array

class OneNearestNeighbor(BaseEstimator, ClassifierMixin):

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
         X, y = check_X_y(X, y)
         self.classes_ = np.unique(y)
         check_classification_targets(self.classes_)
         # XXX fix
         self.sample_ = X
         self.sampleX_ = X
         self.y_ = y
         return self

     def predict(self, X):
         check_is_fitted(self, ["classes_"])
         X = check_array(X)
         y_pred = np.full(shape=len(X), fill_value=self.classes_[0],
                          dtype=self.classes_.dtype)
         # XXX fix
         for i in range(0, len(y_pred)):
             distances = np.sqrt(np.sum(np.square(X[i] - self.sample_),
             distances = np.sqrt(np.sum(np.square(X[i] - self.sampleX_),
                                 axis=-1, keepdims=True))
             y_pred[i] = self.y_[np.argmin(distances)]
         return y_pred

     def score(self, X, y):
         X, y = check_X_y(X, y)
         y_pred = self.predict(X)
         return np.mean(y_pred == y) #pb exit(1)
