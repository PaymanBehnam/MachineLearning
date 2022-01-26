import numpy as np

class PCA(object):

    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def fit(self, X): # 5 points
        """
        Decompose dataset into principal components.
        You may use your SVD function from the previous part in your implementation or numpy.linalg.svd function.

        Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
        corresponding values from PCA.

        Args:
            X: N*D array corresponding to a dataset
        Return:
            None
        """
        XShifted = X - X.mean(axis = 0, keepdims = True)
        self.U, self.S, self.V =  np.linalg.svd(XShifted, full_matrices= False)


    def transform(self, data, K=2): # 2 pts
        """
        Transform data to reduce the number of features such that final data has given number of columns

        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: N*D array corresponding to a dataset
            K: Int value for number of columns to be kept
        Return:
            X_new: N*K array corresponding to data obtained by applying PCA on data
        """
        transform = np.matmul(data , np.transpose(self.V[:K,:])) 
        return transform

    def transform_rv(self, data, retained_variance=0.99): # 3 pts
        """
        Transform data to reduce the number of features such that a given variance is retained

        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: N*D array corresponding to a dataset
            retained_variance: Float value for amount of variance to be retained
        Return:
            X_new: N*K array corresponding to data obtained by applying PCA on data
        """
        variance = self.S ** 2/ np.sum (self.S ** 2)
        index = 0
        obtained_variance = 0
        while (obtained_variance < retained_variance) :
            obtained_variance  =  obtained_variance  + variance[index]
            index = index + 1
        return self.transform(data,index)
        

    def get_V(self):
        """ Getter function for value of V """
        
        return self.V