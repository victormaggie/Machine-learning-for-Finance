import numpy as np

np.random.seed(19890516)

class BagLearner(object):

    def __init__(self, learner, kwargs, bags=20, boost=False, verbose=False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.learners = []

        for _ in range(self.bags):
            self.learners.append(learner(**kwargs))

    def author(self):
        return "rren34"

    def addEvidence(self, dataX, dataY):
        """
        :param self:
        :param dataX:
        :param dataY:
        :return:
        """
        for learner in self.learners:
            indices = np.arange(dataX.shape[0])
            bootstrap = np.random.choice(indices, size=dataX.shape[0], replace=True)
            Xtrain = dataX[bootstrap, :]
            Ytrain = dataY[bootstrap]
            learner.addEvidence(Xtrain, Ytrain)

    def query(self, points):

        votes = []
        for learner in self.learners:
            votes.append(learner.query(points))

        return np.vstack(votes).mean(axis=0).reshape(-1)


if __name__ == "__main__":
    print('well done')
