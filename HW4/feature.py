import numpy as np


def create_nl_feature(X):
    '''
    TODO - Create additional features and add it to the dataset
    
    returns:
        X_new - (N, d + num_new_features) array with 
                additional features added to X such that it
                can classify the points in the dataset.
    '''
    nl_feature = np.zeros([X.shape[0], X.shape[1] + 1])
    nl_feature[:, X.shape[1]] = np.apply_along_axis(np.linalg.norm, 1, X)
    return nl_feature
