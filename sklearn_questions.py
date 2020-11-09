# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Class ONN.

    Own Implementation of OneNearestNeighbor Classifier.
    """

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Class Method for fitting.

        Return the parameters we need for predicting a test set.
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        # XXX fix
        self.data_ = X
        self.labels_ = y

        return self

    def predict(self, X):
        """Class Method for predicting.

        Predicting labels for the test sample.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        # XXX fix
        for j in range(len(X)):
            minn = np.linalg.norm(X[j]-self.data_[0])
            for i in range(1, len(self.data_)):
                distance = np.linalg.norm(X[j]-self.data_[i])
                if distance <= minn:
                    minn = distance
                    label_j = self.labels_[i]
            y_pred[j] = label_j

        return y_pred

    def score(self, X, y):
        """Class method for Scoring.

        Scoring the test data.
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


# Following code is to test our classifier for the iris dataset from 'sklearn'
"""
from sklearn import datasets
iris = datasets.load_iris()
iris_data = iris.data
iris_labels = iris.target
np.random.seed(42)
indices = np.random.permutation(len(iris_data))
n_training_samples = 12
learnset_data = iris_data[indices[:-n_training_samples]]
learnset_labels = iris_labels[indices[:-n_training_samples]]
testset_data = iris_data[indices[-n_training_samples:]]
testset_labels = iris_labels[indices[-n_training_samples:]]

Classifier=OneNearestNeighbor()
Classifier.fit(learnset_data,learnset_labels)
Classifier.predict(testset_data)
print(Classifier.score(testset_data,testset_labels))
"""
