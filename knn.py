import numpy as np
from tqdm import tqdm
import operator
import time
import itertools
from datetime import datetime

class Knn(object):

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def _square_distance(self, v1, v2):
        return np.sum(np.square(v1 - v2))

    def predict(self, X_test):

        # TODO Predict the label of X by
        # the k nearest neighbors.

        # Input:
        # X: np.array, shape (n_samples, n_features)

        # Output:
        # y: np.array, shape (n_samples,)

        # Hint:
        # 1. Use self.X and self.y to get the training data.
        # 2. Use self.k to get the number of neighbors.
        # 3. Use np.argsort to find the nearest neighbors.

        # YOUR CODE HERE
        y_pred=[]
        for i in tqdm(range(((len(X_test))))):
            time.sleep(0.0000001)
            dist_arr = [self._square_distance(X_test[i], self.X[j]) for j in range(len(self.X))]
            sorted_index = np.argsort(dist_arr)
            top_k_index = sorted_index[:self.k]
            '''vote_dictionary = {}
            for y in self.y[top_k_index]:
                if y not in vote_dictionary.keys():
                    vote_dictionary[y] = 1
                else:
                    vote_dictionary[y] += 3
            sorted_vote_dict = sorted(vote_dictionary.items(), key=operator.itemgetter(1), reverse=True)
            y_pred.append(sorted_vote_dict[0][0])'''
            bincount=np.bincount
            maxlabel=bincount(self.y[top_k_index]).argmax()
            y_pred.append(maxlabel)
        return np.array(y_pred)



        # raise NotImplementedError
        ...

        # End of todo
