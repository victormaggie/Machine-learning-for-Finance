import numpy as np
from scipy.stats import pearsonr

class DTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):

        if leaf_size < 1:
            raise Exception("leaf_size is too small cannot be used")
        self.leaf_size = leaf_size

    def author(self):
        return "rren34"

    def build_tree(self, dataX, dataY):           # greedy approach for the split method
        if self.leaf_size >= dataX.shape[0]:  # minimum number of node reached, no need to split
            return np.asarray([[-1, np.mean(dataY), np.nan, np.nan]])

        if np.unique(dataY).shape[0] == 1:    # if all the Y value is the same, no need to split
            return np.asarray([[-1, dataY[0], np.nan, np.nan]])

        correlation = []    # correlation list
        for i in range(dataX.shape[1]):
            cor = np.var(dataX[:, i])     # check if covariance is zero. covariance > 0
            corr_val = np.corrcoef(dataX[:, i], dataY)[0, 1] if cor > 0 else 0
            correlation.append(corr_val)

        # filter out the maximum value
        best_index = np.argsort(correlation)[::-1][0]

        split_val = np.median(dataX[:, best_index])
        is_left_tree = dataX[:, best_index] <= split_val              # return boolean, and slice the data

        if np.median(dataX[is_left_tree, best_index]) == split_val:   # there might have a bug
            return np.asarray([[-1, np.mean(dataY), np.nan, np.nan]])

        # build the left tree
        left_tree = self.build_tree(dataX[is_left_tree], dataY[is_left_tree])

        # build the right tree
        right_tree = self.build_tree(dataX[~is_left_tree], dataY[~is_left_tree])

        # build root
        root = np.asarray([[best_index, split_val, 1, left_tree.shape[0]+1]])

        return np.vstack((root, left_tree, right_tree))

    # define a query tree function
    def query_tree(self, tree, dataX):
        root = tree[0]

        if int(root[0]) == -1:
            # this is a leaf, we return its split value
            return root[1]

        elif dataX[int(root[0])] <= root[1]:
            # go in the left subtree
            left_tree = tree[int(root[2]):, :]
            return self.query_tree(left_tree, dataX)
        else:
            # go in the right subtree
            right_tree = tree[int(root[3]):, :]
            return self.query_tree(right_tree, dataX)

    def addEvidence(self, dataX, dataY):

        """
        :param dataX factor to train:
        :param dataY value to train:
        :return:
        """
        self.tree = self.build_tree(dataX, dataY)

    def query(self, points):

        val = []
        for X in points:
            val.append(self.query_tree(self.tree, X))

        return np.asarray(val)

    if __name__ == "__main__":
        print("well done")
