
import numpy as np

np.random.seed(19890516)

class RTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        if leaf_size < 1:
            raise Exception('leaf size must be greater than zero')
        self.leaf_size = leaf_size

    def author(self):
        return 'rren34'

    def build_tree(self, dataX, dataY):

        # stop condition 1:
        if self.leaf_size >= dataX.shape[0]:
            return np.asarray([[-1, np.mean(dataY), np.nan, np.nan]])

        # stop condition 2:
        if np.unique(dataY).shape[0] == 1:
            return np.asarray([[-1, dataY[0], np.nan, np.nan]])

        # random selection for recursion
        index = np.random.choice(np.arange(dataX.shape[1]))

        # split value
        split_val = np.median(dataX[:, index])

        # criterion for left tree
        is_left_tree = dataX[:, index] <= split_val

        if np.median(dataX[is_left_tree, index]) == split_val:   # there might have a bug
            return np.asarray([[-1, np.mean(dataY), np.nan, np.nan]])

        # left tree
        left_tree = self.build_tree(dataX[is_left_tree], dataY[is_left_tree])

        # right tree
        right_tree = self.build_tree(dataX[~is_left_tree], dataY[~is_left_tree])

        # root value

        root = np.asarray([[index, split_val, 1, left_tree.shape[0] + 1]])

        return np.vstack((root, left_tree, right_tree))

    def query_tree(self, tree, dataX):
        root = tree[0]

        # leaf
        if int(root[0]) == -1:
            return root[1]

        # right tree
        elif root[1] >= dataX[int(root[0])]:
            left_tree = tree[int(root[2]):, :]
            return self.query_tree(left_tree, dataX)

        else:
            right_tree = tree[int(root[3]):, :]
            return self.query_tree(right_tree, dataX)

    def addEvidence(self, dataX, dataY):

        self.tree = self.build_tree(dataX, dataY)

    def query(self, points):
        val = []
        for X in points:
            val.append(self.query_tree(self.tree, X))
        return np.asarray(val)

    if __name__ == "__main__":
        print('well done')




