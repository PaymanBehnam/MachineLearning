import numpy as np

class ImgCompression(object):
    def __init__(self):
        pass

    def svd(self, X): # [5pts]
        """
        Do SVD. You could use numpy SVD.
        Your function should be able to handle black and white
        images (N*D arrays) as well as color images (N*D*3 arrays)
        In the image compression, we assume that each column of the image is a feature. Image is the matrix X.
        Args:
            X: N * D array corresponding to an image (N * D * 3 if color image)
        Return:
            U: N * N for black and white images / N * N * 3 for color images
            S: min(N, D) * 1 for black and white images / min(N, D) * 3 for color images
            V: D * D for black and white images / D * D * 3 for color images
        """

        if len(X.shape) > 2:
            U = []
            V = []
            S = []
            for i in range(len(X.shape)):
                TempU, TempS, TempV = np.linalg.svd(X[:, :, i])
                U.append(TempU)
                S.append(TempS)
                V.append(TempV)
            U = np.stack(U, axis=-1)
            V = np.stack(V, axis=-1)
            S = np.stack(S, axis=-1)
        else:
            U, S, V = np.linalg.svd(X)
        return U, S, V

    def rebuild_svd(self, U, S, V, k): # [5pts]
        """
        Rebuild SVD by k componments.
        Args:
            U: N*N (*3 for color images)
            S: min(N, D)*1 (*3 for color images)
            V: D*D (*3 for color images)
            k: int corresponding to number of components
        Return:
            Xrebuild: N*D array of reconstructed image (N*D*3 if color image)

        Hint: numpy.matmul may be helpful for reconstructing color images
        """
        D=V.shape[0]
        N=U.shape[0]
        if(len(U.shape)==2):
            rebuild_svd=np.zeros((N,D))
            for i in range(k):
                rebuild_svd += S[i] * np.matmul(U[:,i].reshape(N,1),V[i,:].reshape(1,D))
            return rebuild_svd
        else:
            rebuild_svd=np.zeros((N,D,3))
            for j in range(3):
                for i in range(k):
                    rebuild_svd[:,:,j] += S[i,j] * np.matmul(U[:,i,j].reshape(N,1),V[i,:,j].reshape(1,D))
            return rebuild_svd

        

    def compression_ratio(self, X, k): # [5pts]
        """
        Compute compression of an image: (num stored values in compressed)/(num stored values in original)
        Args:
            X: N * D array corresponding to an image (N * D * 3 if color image)
            k: int corresponding to number of components
        Return:
            compression_ratio: float of proportion of storage used by compressed image
        """
        Xshape0= X.shape[0]
        Xshape1 = X.shape[1]
        compression_ratio = ((Xshape0 + Xshape1 + 1) * k) / (Xshape1 * Xshape0)
        return compression_ratio

    def recovered_variance_proportion(self, S, k): # [5pts]
        """
        Compute the proportion of the variance in the original matrix recovered by a rank-k approximation

        Args:
           S: min(N, D)*1 (*3 for color images) of singular values for the image
           k: int, rank of approximation
        Return:
           recovered_var: float (array of 3 floats for color image) corresponding to proportion of recovered variance
        """
        #if S.shape[1] > 2:
        #if len(S.shape) > 2:
         #   recovered_variance_proportion = np.sum(S[0:k] ** 2, axis=0) / np.sum(S ** 2, axis=0)
        #else:
        #    recovered_variance_proportion = np.sum(S[0:k] ** 2) / np.sum(S ** 2)
        #return recovered_variance_proportion
        S = S.reshape([S.shape[0], -1])
        recovered_variance_proportion = np.zeros(S.shape[1])
        Spow2 = S ** 2
        for index in range (S.shape[1]):
               recovered_variance_proportion[index] =  np.sum(Spow2[:k, index]) / np.sum(Spow2[:, index])
        return recovered_variance_proportion     


        
