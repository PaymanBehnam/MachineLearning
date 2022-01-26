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

# Set random seed so output is all same
np.random.seed(1)

def pairwise_dist (x, y):
    """
    Args:
        x: N x D numpy array
        y: M x D numpy array
    Return:
        dist: N x M array, where dist2[i, j] is the euclidean distance between
        x[i, :] and y[j, :]
    """
    tempy0=y.shape[0]
    tempx0=x.shape[0]
    tempy1=y.shape[1]
    x=x.reshape(tempx0,1,tempy1)
    x=np.repeat(x, tempy0, 1)
    y=y.reshape(1,tempy0,tempy1)
    z=np.sum(np.power(np.subtract(x,y),2), axis = 2)
    z=np.sqrt(z)
    return z
class KMeans(object):

    def __init__(self): #No need to implement
        pass
    
    
    def _init_centers(self, points, K, **kwargs):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            kwargs: any additional arguments you want
        Return:
            centers: K x D numpy array, the centers. (Initialize the centers at random data point coordinates.)
        """
       
        centers = points[points.shape[0]-K-1:points.shape[0]-1, :]
        return centers

    def _update_assignment(self, centers, points):
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            points: NxD numpy array, the observations
        Return:
            cluster_idx: numpy array of length N, the cluster assignment for each point

        Hint: You could call pairwise_dist() function.
        """
        cluster_update = pairwise_dist(points, centers)
        cluster_update = np.argmin(cluster_update, axis=1)
        return cluster_update

    def _update_centers(self, old_centers, cluster_idx, points):
        """
        Args:
            old_centers: old centers KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            centers: new centers, K x D numpy array, where K is the number of clusters, and D is the dimension.
        """
        temp_p0=points.shape[0]
        temp_oc0=old_centers.shape[0]
        temp_p0oc0 = np.zeros((temp_p0, temp_oc0))
        index=np.arange(temp_p0)
        temp_p0oc0[index,cluster_idx.reshape(-1)]=1
        sums=np.sum(temp_p0oc0,axis=0)
        points=points.transpose()
        update_centers=np.matmul(points,temp_p0oc0)
        update_centers=update_centers/sums
        update_centers=update_centers.transpose()
        return update_centers

    def _get_loss(self, centers, cluster_idx, points):
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            loss: a single float number, which is the objective function of KMeans.
        """
        distance = np.power(pairwise_dist(centers, points), 2)
        loss = distance[cluster_idx, np.arange(len(points))]
        return np.sum(loss)
        
    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, verbose=False, **kwargs):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            verbose: boolean to set whether method should print loss (Hint: helpful for debugging)
            kwargs: any additional arguments you want
        Return:
            cluster assignments: Nx1 int numpy array
            cluster centers: K x D numpy array, the centers
            loss: final loss value of the objective function of KMeans
        """
        centers = self._init_centers(points, K, **kwargs)
        for it in range(max_iters):
            cluster_idx = self._update_assignment(centers, points)
            centers = self._update_centers(centers, cluster_idx, points)
            loss = self._get_loss(centers, cluster_idx, points)
            K = centers.shape[0]
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            if verbose:
                print('iter %d, loss: %.4f' % (it, loss))
        return cluster_idx, centers, loss

def find_optimal_num_clusters(data, max_K=19):  # [10 pts]
        np.random.seed(1)
        """Plots loss values for different number of clusters in K-Means

        Args:
            data: input data of shape NxD
            max_K: number of clusters
        Return:
            List with loss values
        """
        z = []
        for index in range(max_K):
            cluster_idx, centers, lossy = KMeans()(data, index + 1)
            z.append(lossy)
        plt.plot(z)
        plt.show()
        return z
def intra_cluster_dist(cluster_idx, data, labels): # [4 pts]
        """
        Calculates the average distance from a point to other points within the same cluster

        Args:
            cluster_idx: the cluster index (label) for which we want to find the intra cluster distance
            data: NxD numpy array, where N is # points and D is the dimensionality
            labels: 1D array of length N where each number indicates of cluster assignement for that point
        Return:
            intra_dist_cluster: 1D array where the i_th entry denotes the average distance from point i
                                in cluster denoted by cluster_idx to other points within the same cluster
        """
        data_labels_cluster_idx = data[labels == cluster_idx,:]
        pops = data_labels_cluster_idx.shape[0]
        #distance = pairwise_dist(data_labels_cluster_idx, data)
        distance = pairwise_dist(data_labels_cluster_idx, data_labels_cluster_idx)

        update_distance = np.nan_to_num(distance)
        if pops == 0  or pops == 1:
            return 1
        return   np.sum(update_distance, axis =1)/(pops-1)
        

def inter_cluster_dist(cluster_idx, data, labels): # [4 pts]
        """
        Calculates the average distance from one cluster to the nearest cluster
        Args:
            cluster_idx: the cluster index (label) for which we want to find the intra cluster distance
            data: NxD numpy array, where N is # points and D is the dimensionality
            labels: 1D array of length N where each number indicates of cluster assignement for that point
        Return:
            inter_dist_cluster: 1D array where the i-th entry denotes the average distance from point i in cluster
                                denoted by cluster_idx to the nearest neighboring cluster
        """
        sum_of_clusters = np.max(labels)+1
        data_labels_cluster_idx = data[labels == cluster_idx,:]
        pops = data_labels_cluster_idx.shape[0]
        
        distance = pairwise_dist(data_labels_cluster_idx, data)
        inter_cluster_distances = np.zeros([pops, sum_of_clusters-1])
        
        counter = 0
        for j in range(sum_of_clusters):
            if j != cluster_idx:
                pop_clusterj = data[labels == j, :].shape[0]
                inter_cluster_distances[:, counter]= np.sum(distance[:,labels == j], axis=1)/pop_clusterj
                counter = counter+1
        return np.min(inter_cluster_distances, axis=1)
        


def silhouette_coefficient(data, labels):  # [2 pts]
        """
        Finds the silhouette coefficient of the current cluster assignment

        Args:
            data: NxD numpy array, where N is # points and D is the dimensionality
            labels: 1D array of length N where each number indicates of cluster assignement for that point
        Return:
            silhouette_coefficient: Silhouette coefficient of the current cluster assignment
        """
        sum_of_points = data.shape[0]
        sum_of_clusters = np.max(labels) + 1

        s_sum = 0
        for i in range (sum_of_clusters):
            a_i  = intra_cluster_dist(i, data, labels)
            b_i  = inter_cluster_dist(i, data, labels)
            s_i = (b_i - a_i)/np.maximum(b_i, a_i)
            s_sum = s_sum + np.sum(s_i)
        return s_sum/sum_of_points




if __name__ == '__main__':

    from kmeans import pairwise_dist

    x = np.random.randn(2, 2)
    y = np.random.randn(3, 2)

    print("*** Expected Answer ***")
    print("""==x==
    [[ 1.62434536 -0.61175641]
    [-0.52817175 -1.07296862]]
    ==y==
    [[ 0.86540763 -2.3015387 ]
    [ 1.74481176 -0.7612069 ]
    [ 0.3190391  -0.24937038]]
    ==dist==
    [[1.85239052 0.19195729 1.35467638]
    [1.85780729 2.29426447 1.18155842]]""")


    print("\n*** My Answer ***")
    print("==x==")
    print(x)
    print("==y==")
    print(y)
    print("==dist==")
    print(pairwise_dist(x, y))
