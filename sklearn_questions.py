# noqa: D100
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.multiclass import check_classification_targets


class OneNearestNeighbor(BaseEstimator, ClassifierMixin):
    """predection of the nearest neighbor to a given data
    """
    def __init__(self):  # noqa: D107
        pass

    def fit(self, X, y):
        """fitting the data
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_=X
        self.y_=y
        
        
        # XXX fix
        
        return self

    def predict(self, X):
        """Class prediction
        """
        # check_is_fitted(self)
        # X = check_array(X)
        
        # y_pred = np.full(shape=len(X),dtype=self.classes_.dtype,fill_value=self.classes_[0])
        #y_pred = np.full(shape=len(X),fill_value=self.classes_[0])

        ########################## Manual method #################
        # def dist_euc(x,y):
        #     dist=0
        #     for j in range(len(x)):
        #           dist+=(x[j]-y[j])**2
        #     return(dist**0.5)
        
        # for i in range (X.shape[0]) :
        #     distance=[]
        #     x=X[i,:]
        #     train=self.X_
        #     for tr in train:
        #         if not(all(tr == x)) :
        #             distance.append((tr,dist_euc(x,tr)))
        #             distance.sort(key=lambda tup: tup[1])
        #             index=0
        #             neib=distance[0][0]
        #             for e in range(self.X_.shape[0]):
        #                 if all(self.X_[e,:]==neib):
        #                     index=e
        #             y_pred[i]=self.y_[index]
        
        #################With pre-defined functions###########################
        
        check_is_fitted(self)
        X = check_array(X)
        # dist=euclidean_distances( self.X_,X)
        dist=np.zeros((len(X),len(self.X_)))
        for i in range(dist.shape[0]):
            for j in range(dist.shape[1]):
                dist[i,j]=np.linalg.norm(X[i]-self.X_[j],2)
        
        
        neib=np.argmin(dist,axis=1)
        y_pred = np.full(shape=len(X), dtype=self.classes_.dtype,fill_value=self.y_[neib])
        
        
        # XXX fix
        return y_pred
    

    def score(self, X, y):
        """Score calculationg
        """
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
