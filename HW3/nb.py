import numpy as np
from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split
import pandas as pd

RANDOM_SEED = 5


class NaiveBayes(object):
    def __init__(self):
        pass

    def likelihood_ratio(self, X_republican, X_democratic):  # [5pts]
        '''
        Args:
            X_republican: N_republican x D where N_republican is the number of republican tweets that we have,
                while D is the number of features (we use the word count as the feature)
            X_democratic: N_democratic x D where N_democratic is the number of democratic tweets that we have,
                while D is the number of features (we use the word count as the feature)
        Return:
            likelihood_ratio: 2 x D matrix of the likelihood ratio of different words for different class of tweets
        '''
        d = X_republican.shape[1]
        likelihood_ratio = np.ones([2, d])

        likelihood_ratio[0, :] = (1 + np.sum(X_democratic, axis=0)) / (d+np.sum(X_democratic))
        likelihood_ratio[1, :] = (1 + np.sum(X_republican, axis=0)) / (d+np.sum(X_republican))


        return likelihood_ratio

    def priors_prob(self, X_republican, X_democratic):  # [5pts]
        '''
        Args:
            X_republican: N_republican x D where N_republican is the number of republican tweets that we have,
                while D is the number of features (we use the word count as the feature)
            X_democratic: N_democratic x D where N_democratic is the number of democratic tweets that we have,
                while D is the number of features (we use the word count as the feature)
        Return:
            priors_prob: 1 x 2 matrix where each entry denotes the prior probability for each class
        '''
        priors_prob_repub = np.sum(X_republican)/(np.sum(X_democratic)+np.sum(X_republican))
        priors_prob_democ = np.sum(X_democratic)/(np.sum(X_democratic)+np.sum(X_republican))
        return [priors_prob_repub, priors_prob_democ]

    # [5pts]
    def analyze_affiliation(self, likelihood_ratio, priors_prob, X_test):
        '''
        Args:
            likelihood_ratio: 2 x D matrix of the likelihood ratio of different words for different class of tweets
            priors_prob: 1 x 2 matrix where each entry denotes the prior probability for each class
            X_test: N_test x D bag of words representation of the N_test number of tweets that we need to analyze its political affiliation
        Return:
             1 x N_test list, each entry is a class label indicating the tweet's political affiliation (republican: 0, democratic: 1)
        '''
      


        probability = np.zeros([X_test.shape[0], 2])
        
        probability[:, 1] = (X_test ** likelihood_ratio[1, :].reshape(1, -1)).prod(axis=1) * priors_prob[1]
        probability[:, 0] = (X_test ** likelihood_ratio[0, :].reshape(1, -1)).prod(axis=1) * priors_prob[0]

        label = np.zeros([X_test.shape[0], 1])
        prob_1 = probability[:, 1]
        prob_0 = probability[:, 0]
        bool_index = prob_1 > prob_0
        label[bool_index] = 1
        label = np.transpose(label)
        return label
