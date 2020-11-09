# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.metrics.pairwise import euclidean_distances


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """Classe de modèle OneNearestNeighbor qui est une particularité 
    du KNearestNeighbor pour K=1.
    
    Attributes :
        
      X_classes : possible classes 
      X_train : features of the training set characterizing the model
      y_train : classes of the training set characterizing the model

        
    """
    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """ Fitting of the model : the model is entirely determined by the 
        training database. So the fitting consists in the stocking of X and y .
        We define the possibles y classes.
               
        Parameters : 
        -------
        X : ndarray of shape (n_samples, n_features)
        Features values of the training examples.
        
        y : ndarray of shape (n_samples, 1)
        Classes of the training examples.
        
        Returns
        -------
        
        self: le modèle 
        """
        
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.X_train_ = X
        self.y_train_ = y
    
        return self

    def predict(self, X):
        """
        Prediction of the classes of the exemples in the ndarray X.
        For every example, the prediction is the class of the closest
        training example with regard to the euclidian distance.
        
        
        Parameters : 
        -------
        X : ndarray of shape (n_samples, n_features)
        The input array.
        

        Returns
        -------
        
        y_pred : ndarray of shape (n_samples, 1)
        The output array.             
        """
        
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.full(shape=len(X), fill_value=self.classes_[0])
        distances = euclidean_distances(X, self.X_train_)
        index_closest = np.argmax(distances, axis=1)
        y_pred = self.y_train_[index_closest, 1]     
                   
        return y_pred

    def score(self, X, y):
        """ Performance of the model self on a set of examples (X,y).
        The metrics used is : 
        number of errors/total number of examples.
        
        Parameters : 
        -------
        X : ndarray of shape (n_samples, n_features)
        Features of the examples.
         
        y : ndarray of shape (n_samples, 1)
        Classes of the examples.
        
        Returns
        -------
        
        score : number of errors/total number of examples (float).     
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        
        return np.mean(y_pred == y)
