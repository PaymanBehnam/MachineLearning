from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm
# Load image
import imageio


SIGMA_CONST = 1e-6
LOG_CONST = 1e-32

class GMM(object):
    def __init__(self, X, K, max_iters = 100): # No need to change
        """
        Args: 
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters
        
        self.N = self.points.shape[0]        #number of observations
        self.D = self.points.shape[1]        #number of features
        self.K = K                           #number of components/clusters

    #Helper function for you to implement 
    def softmax(self,logits):
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        """
        update_logits = (logits.transpose() - np.amax(logits, axis=1))
        update_logits = np.exp(update_logits)
        update_logits = update_logits / np.sum(update_logits, axis=0)
        return update_logits.transpose()

    def logsumexp(self,logits):
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        """
        maximum=np.max(logits,axis=1).reshape(logits.shape[0],-1)
        update_logits=np.subtract(logits,maximum)
        update_logits=np.exp(update_logits)
        sums=np.sum(update_logits,axis=1).reshape(update_logits.shape[0],-1)
        logsums=np.log(sums)
        return np.add(logsums,maximum)
    def _init_components(self, **kwargs):
        """
        Args:
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. 
                You will have KxDxD numpy array for full covariance matrix case

        Hint: You can use KMeans to initialize the centers for each gaussian
        """
        temp_p1=self.points.shape[1]
        pi=np.ones(self.K)/self.K
        mu=np.ones((self.K,temp_p1))
        sigma=np.ones((self.K,temp_p1,temp_p1))
        return pi,mu,sigma
    #for grad students
    def multinormalPDF(self, logits, mu_i, sigma_i):  #[5pts]
        """
        Args: 
            logit: N x D numpy array
            mu_i: 1xD numpy array, the center for the ith gaussian.
            sigma_i: 1xDxD numpy array, the covariance matrix of the ith gaussian.  
        Return:
            normal_pdf: (N,) numpy array, the probability distribution of N data for the ith gaussian
            
        Hint: 
            np.linalg.det() and np.linalg.inv() should be handy.
            Add SIGMA_CONST if you encounter LinAlgError
        """
        """pdf of the multivariate normal distribution."""
        # m = len(mu_i)
        # sigma2 = np.diag(sigma_i)
        # X = logits-mu_i.T
        # p = 1/((2*np.pi)**(m/2)*np.linalg.det(sigma2)**(0.5))*np.exp(-0.5*np.sum(X.dot(np.linalg.pinv(sigma2))*X,axis=1))

        # return p
        logits_deviated = logits - mu_i.reshape(1,-1)
        logits_deviated_mul = np. matmul(logits_deviated, np.linalg.inv(sigma_i + 1e-32 * np.eye(sigma_i.shape[1])))
        #try:
        multinormalPDF  = 1/(2*np.pi)**(len(logits[0])/2)*np.linalg.det(sigma_i)**(-1/2)*np.exp(((-1/2)*logits_deviated_mul * logits_deviated).sum(axis=1, keepdims=True))
        #except:
        #    print("An exception occurred") 
        multinormalPDF = multinormalPDF.reshape(-1)

        return multinormalPDF 

    def _ll_joint(self, pi, mu, sigma, full_matrix = True, **kwargs):
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
            
        Return:
            ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
            
        Hint: You might find it useful to add LOG_CONST to the expression before taking the log 
        """
        # lenpi = len(pi)
        # lenp0 = len(self.points[0])
        # lenpoints = len(self.points)
        # ll = np.zeros((lenpoints, lenpi))
        # sigma = sigma + 1e-32
        # for i in range(lenpi):
        #     x = self.points - mu[i]
        #     x = np.exp(-0.5 * np.einsum('ij,ij->i', (x.dot(np.linalg.pinv(sigma[i]))), x))
        #     ll[:, i] = np.log(pi)[i] + np.log(1.0 / (np.sqrt((2 * np.pi) ** lenp0 * np.linalg.det(sigma[i]))) * x)
        # return ll

        ll =[]
    
        for index in range(self.K):
            #ll.append(np.log(pi[index]) + np.log(self.multinormalPDF(self.points, mu[index], sigma[index])))
            ll.append(np.log(pi[index] + LOG_CONST) + np.log(self.multinormalPDF(self.points, mu[index], sigma[index]) + LOG_CONST))

        return np.transpose(ll)

       

    def _E_step(self, pi, mu, sigma, **kwargs):
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            
        Hint: 
            You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above. 
        """
        return self.softmax(self._ll_joint( pi, mu, sigma, **kwargs))
       
        

    def _M_step(self, gamma, full_matrix = True, **kwargs):
        """
        Args:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            
        Hint:  
            There are formulas in the slide and in the above description box.
        """
        sum_gamma = np.sum(gamma, axis=0)
        lenp = len(self.points)
        lenp0 = len(self.points[0])
        leng0 = len(gamma[0])
        pi = sum_gamma / lenp
        mu = np.zeros((leng0, lenp0))
        sigma = np.zeros((leng0, lenp0, lenp0))
        for i in range(leng0):
            mu[i] = np.matmul(gamma[:, i].transpose(), self.points) / sum_gamma[i]
            sigma[i] = 1 / sum_gamma[i] * np.einsum('i,ij->ij', gamma[:, i], (self.points - mu[i])).T.dot(self.points - mu[i])
        return pi, mu, sigma

    def __call__(self, full_matrix = True, abs_tol=1e-16, rel_tol=1e-16, **kwargs): # No need to change
        """
        Args:
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want
        
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)       
        
        Hint: 
            You do not need to change it. For each iteration, we process E and M steps, then update the paramters. 
        """
        pi, mu, sigma = self._init_components(**kwargs)
        pbar = tqdm(range(self.max_iters))
        
        for it in pbar:
            # E-step
            gamma = self._E_step(pi, mu, sigma)
            
            # M-step
            pi, mu, sigma = self._M_step(gamma, full_matrix)
            
            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return gamma, (pi, mu, sigma)
