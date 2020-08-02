import BagLearner
import LinRegLearner

class InsaneLearner(object):

    def __init__(self, verbose=False):
        self.learner = BagLearner.BagLearner(learner=BagLearner.BagLearner, kwargs={"learner": LinRegLearner.LinRegLearner, "kwargs": {}, "bags": 20, "boost": False, "verbose": verbose},\
                                             bags=20, boost=False, verbose=verbose)

    def author(self):
        return 'rren34'

    def addEvidence(self, dataX, dataY):
        self.learner.addEvidence(dataX, dataY)

    def query(self, data_points):
        return self.learner.query(data_points)