import numpy as np


class Regression(object):

    def __init__(self):
        pass

    def rmse(self, pred, label):  # [5pts]
        '''
        This is the root mean square error.
        Args:
            pred: numpy array of length N * 1, the prediction of labels
            label: numpy array of length N * 1, the ground truth of labels
        Return:
            a float value
        '''
        erSq = (pred-label)**2
        erMean = np.sum(erSq) / erSq.shape[0]
        rmse = erMean**(1/2)

        return rmse

    def construct_polynomial_feats(self, x, degree):  # [5pts]
        """
        Args:
            x: N x D numpy array, where N is number of instances and D is the
               dimensionality of each instance.
            degree: the max polynomial degree.
        Return:
            feat: numpy array of shape N x (degree+1) x D, remember to include
                  the bias term.

            Example: print(feat)
            For an input where N=3, D=2, and degree=3...

            [[[ 1.0        1.0]
              [ x_{1,1}    x_{1,1}]
              [ x_{1,1}^2  x_{1,2}^2]
              [ x_{1,1}^3  x_{1,2}^3]]

             [[ 1.0        1.0]
              [ x_{2,1}    x_{2,2}]
              [ x_{2,1}^2  x_{2,2}^2]
              [ x_{2,1}^3  x_{2,2}^3]]

             [[ 1.0        1.0]
              [ x_{3,1}    x_{3,2}]
              [ x_{3,1}^2  x_{3,2}^2]
              [ x_{3,1}^3  x_{3,2}^3]]]
        """
        #N, D = x.shape
        xndim = x.ndim
        if xndim == 1:
            x = x.reshape(-1,1)
        
        #feat = np.ones((N, D, (degree+1)))
        feat = np.ones((x.shape[0], (degree+1), x.shape[1]))


        for i in range(degree+1):
            feat[:,i,:] = x**i


        if xndim == 1:
            feat = feat.reshape(-1, degree+1)
        
        return feat

    def predict(self, xtest, weight):  # [5pts]
        """
        Args:
            xtest: NxD numpy array, where N is number
                   of instances and D is the dimensionality of each
                   instance
            weight: Dx1 numpy array, the weights of linear regression model
        Return:
            prediction: Nx1 numpy array, the predicted labels
        """
        predict = np.matmul(xtest,weight)
        return predict

    def linear_fit_closed(self, xtrain, ytrain):  # [5pts]
        """
        Args:
            xtrain: N x D numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: N x 1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        
        xtemp = np.linalg.pinv(xtrain)
        return np.matmul(xtemp, ytrain)

    def linear_fit_GD(self, xtrain, ytrain, epochs=5, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        N, D = xtrain.shape

        weight = np.zeros((D, 1))
        
        
        for i in range(epochs):

            errTemp1 = ytrain - self.predict(xtrain, weight)
            errTemp2 = N * (np.dot(xtrain.T, errTemp1))
            weight = weight + learning_rate/N * errTemp2

        return weight

    def linear_fit_SGD(self, xtrain, ytrain, epochs=100, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        N, D = xtrain.shape
        weight = np.zeros((D, 1))
        
        

        for i in range(epochs):
            for j in range(N):

             
                errTemp1 = ytrain[j] - self.predict(xtrain[j], weight)
                weight += learning_rate * xtrain[j].reshape(D,1) * errTemp1
                
        
        return weight



    def ridge_fit_closed(self, xtrain, ytrain, c_lambda):  # [5pts]
        """
        Args:
            xtrain: N x D numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: N x 1 numpy array, the true labels
            c_lambda: floating number
        Return:
            weight: Dx1 numpy array, the weights of ridge regression model
        """
        N, D = xtrain.shape

        xTemp = np.transpose(xtrain)
        return np.linalg.inv(xTemp @ xtrain + c_lambda * np.identity(D)) @ xTemp @ ytrain


    def ridge_fit_GD(self, xtrain, ytrain, c_lambda, epochs=500, learning_rate=1e-7):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
            c_lambda: floating number
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        N, D = xtrain.shape
        weight = np.zeros((D, 1))
        
        
        for i in range(epochs):

           
            errTemp = ytrain - self.predict(xtrain, weight)
            grad =  np.dot(xtrain.T, errTemp) + c_lambda * np.linalg.norm(weight)
            weight +=  (learning_rate / N) * grad
        return weight

    def ridge_fit_SGD(self, xtrain, ytrain, c_lambda, epochs=100, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        N, D = xtrain.shape
        weight = np.zeros((D, 1))

        for i in range(epochs):
            for j in range(N):

                errTemp = ytrain[j] - xtrain[j, :] @ weight

                MulerrTemp = xtrain[j, :] * errTemp
                gradient = MulerrTemp[:, np.newaxis] + 2 * c_lambda * weight

                weight += learning_rate/N * gradient
        
        return weight

    def ridge_cross_validation(self, X, y, kfold=10, c_lambda=100):  # [8 pts]
        """
        Args:
            X : NxD numpy array, where N is the number of instances and D is the dimensionality of each instance
            y : Nx1 numpy array, true labels
            kfold: Number of folds you should take while implementing cross validation.
            c_lambda: Value of regularization constant
        Returns:
            meanErrors: Float average rmse error
        Hint: np.concatenate might be helpful.
        Look at 3.5 to see how this function is being used.
        # For cross validation, use 10-fold method and only use it for your training data (you already have the train_indices to get training data).
        # For the training data, split them in 10 folds which means that use 10 percent of training data for test and 90 percent for training.
        """

        mean_error = np.zeros(kfold)
        interval = int(len(X) / 10)
        walking = np.arange(len(X))
        for i in range(kfold):
            inferAndis = np.arange(i * interval, (i + 1) * interval, 1)
            trainAndis = np.delete(walking, inferAndis)
            weight = self.ridge_fit_closed(X[trainAndis], y[trainAndis], c_lambda)
            predict = self.predict(X[inferAndis], weight)
            mean_error[i] = self.rmse(predict, y[inferAndis])
        return np.average(mean_error)