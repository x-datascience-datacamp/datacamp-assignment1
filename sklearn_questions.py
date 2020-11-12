"""Assignment - making a sklearn estimator.

The goal of this assignment is to implement by yourself a scikit-learn
estimator for the OneNearestNeighbor and check that it is working properly.

The nearest neighbor classifier predicts for a point X_i the target y_k of
the training sample X_k which is the closest to X_i. We measure proximity with
the Euclidean distance. The model will be evaluated with the accuracy (average
number of samples corectly classified). You need to implement the `fit`,
`predict` and `score` methods for this class. The code you write should pass
the test we implemented. You can run the tests by calling at the root of the
repo `pytest test_sklearn_questions.py`.

We also ask to respect the pep8 convention: https://pep8.org. This will be
enforced with `flake8`. You can check that there is no flake8 errors by
calling `flake8` at the root of the repo.

Finally, you need to write docstring similar to the one in `numpy_questions`
for the methods you code and for the class. The docstring will be checked using
`pydocstyle` that you can also call at the root of the repo.
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """OneNearestNeighbor."""

    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """Fitting function.

         Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            training data of shape
        y : ndarray of shap (n_samples,)
            target values of shape
        Returns
        ----------
        self : OneNearestNeighbor()
               the current instance of the classifier
        """
        X, y = check_X_y(X, y)
        y = check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """Predict function.

        Parameters
        ----------
        X : ndarray of shape (n_test_samples, n_features)
            test data to predict on
        Returns
        ----------
        y : ndarray of shape (n_test_samples,)
            Class labels for each test data sample
        """
        check_is_fitted(self)
        X = check_array(X)
        all_distances = pairwise_distances(X, self.X_)
        closest_dist = np.argmin(all_distances, axis=1)
        y_pred = self.y_[closest_dist]
        return y_pred

    def score(self, X, y):
        """Calculate the Score of the prediction.

        And describe parameters
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return y_pred.mean()
