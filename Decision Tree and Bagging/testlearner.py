"""  		   	  			  	 		  		  		    	 		 		   		 		  
Random Tree Learning!   	  			  	 		  		  		    	 		 		   		 		  
"""

import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as it
import pandas as pd
import numpy as np
import time
import sys
import math
import matplotlib.pyplot as plt

np.random.seed(903474021)

def test():

    if len(sys.argv) != 2:

        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.array([map(float, s.strip().split(',')) for s in inf.readlines()])

    # split 60% as training rows, split 40% as testing rows
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows, 0:-1]
    trainY = data[:train_rows, -1]
    testX = data[train_rows:, 0:-1]
    testY = data[train_rows:, -1]
    print(testX.shape)
    print(testY.shape)

    # create a learner and train it
    # learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner

    learner = dt.DTLearner(leaf_size=1, verbose=False)
    learner.addEvidence(trainX, trainY)  # train it

    print(learner.author())

    # evaluate in sample
    predY = learner.query(trainX)           # root mean square error for training dataset
    rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
    print("In sample results")
    print("RMSE: ", rmse)
    c = np.corrcoef(predY, y=trainY)
    print("corr: ", c[0, 1])

    # evaluate out of sample
    predY = learner.query(testX)            # root mean square error for testing dataset
    rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
    print("Out of sample results")
    print("RMSE: ", rmse)
    c = np.corrcoef(predY, y=testY)
    print("corr: ", c[0, 1])

def data_prep():

    df = pd.read_csv("./Data/Istanbul.csv", header=0)
    X = df.drop(["date", "EM"], axis=1).values
    Y = df["EM"].values
    train_rows = int(0.6 * X.shape[0])
    trainX = X[:train_rows, :]
    trainY = Y[:train_rows]
    testX = X[train_rows:, :]
    testY = Y[train_rows:]
    train_rmse_values = []
    test_rmse_values = []
    return trainX, trainY, testX, testY, train_rmse_values, test_rmse_values

def question_1():
    trainX, trainY, testX, testY, train_rmse_values, test_rmse_values = data_prep()
    leaf_size_values = np.arange(1, 18+1, dtype=np.uint32)
    for leaf_size in leaf_size_values:
        learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)
        learner.addEvidence(trainX, trainY)
        train_predY = learner.query(trainX)
        # RMSE for training error
        train_rmse = math.sqrt(((trainY - train_predY) ** 2).sum() / trainY.shape[0])
        train_rmse_values.append(train_rmse)
        test_predY = learner.query(testX)
        # RMSE for testing error
        test_rmse = math.sqrt(((testY - test_predY) ** 2).sum() / testY.shape[0])
        test_rmse_values.append(test_rmse)
    fig, ax = plt.subplots()
    pd.DataFrame({
        "Train RMSE": train_rmse_values,
        "Test RMSE": test_rmse_values
    }, index=leaf_size_values).plot(
        ax=ax,
        style="o-",
        title="Root Mean Square Error with Leaf Size"
    )

    plt.xticks(leaf_size_values,leaf_size_values)
    plt.xlabel("Leaf size")
    plt.ylabel("RMSE")
    plt.legend(loc=4)
    ax.xaxis.grid(True, which='major', linestyle='dotted')
    ax.yaxis.grid(True, which='major', linestyle='dotted')
    ax.annotate('Overfitting happens', xytext = (11, 0.0045), xy=(10, 0.0052),\
                arrowprops=dict(facecolor='white', shrink=0.05))
    #plt.show()
    plt.tight_layout()
    plt.savefig("Q1.png")

#
def question_2a():
    trainX, trainY, testX, testY, train_rmse_values, test_rmse_values = data_prep()
    leaf_size_values = np.arange(1, 18+1, dtype=np.uint32)
    for leaf_size in leaf_size_values:
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": leaf_size}, bags=20, boost=False, verbose=False)
        learner.addEvidence(trainX, trainY)
        train_predY = learner.query(trainX)
        train_rmse = math.sqrt(((trainY - train_predY) ** 2).sum() / trainY.shape[0])
        train_rmse_values.append(train_rmse)
        test_predY = learner.query(testX)
        test_rmse = math.sqrt(((testY - test_predY) ** 2).sum() / testY.shape[0])
        test_rmse_values.append(test_rmse)
    fig, ax = plt.subplots()
    pd.DataFrame({
        "Train RMSE": train_rmse_values,
        "Test RMSE": test_rmse_values
    }, index=leaf_size_values).plot(
        ax=ax,
        style="o-",
        title="Root Mean Square Error with leaf size for BagLearner",
    )

    plt.xticks(leaf_size_values,leaf_size_values)
    ax.xaxis.grid(True, which='major', linestyle='dotted')
    ax.yaxis.grid(True, which='major', linestyle='dotted')
    plt.xlabel("Leaf size")
    plt.ylabel("RMSE")
    plt.legend(loc=4)
    #plt.show()
    plt.tight_layout()
    plt.savefig("Q2a.png")
#

def question_3a():
    trainX, trainY, testX, testY, train_rmse_values, test_rmse_values = data_prep()
    leaf_size_values = np.arange(1, 18 + 1, dtype=np.uint32)

    for leaf_size in leaf_size_values:
        learner = rt.RTLearner(leaf_size=leaf_size)
        learner.addEvidence(trainX, trainY)
        train_predY = learner.query(trainX)
        train_rmse = math.sqrt(((trainY - train_predY) ** 2).sum() / trainY.shape[0])
        train_rmse_values.append(train_rmse)
        test_predY = learner.query(testX)
        test_rmse = math.sqrt(((testY - test_predY) ** 2).sum() / testY.shape[0])
        test_rmse_values.append(test_rmse)
    fig, ax = plt.subplots()
    pd.DataFrame({
        "Train RMSE": train_rmse_values,
        "Test RMSE": test_rmse_values
    }, index=leaf_size_values).plot(
        ax=ax,
        style="o-",
        title="Root Mean Square Error with Leaf Size for RTLearner"
    )
    plt.xticks(leaf_size_values, leaf_size_values)
    plt.xlabel("Leaf size")
    plt.ylabel("RMSE")
    ax.xaxis.grid(True, which='major', linestyle='dotted')
    ax.yaxis.grid(True, which='major', linestyle='dotted')
    plt.legend(loc=4)
    plt.tight_layout()
    #plt.show()
    plt.savefig("Q3a.png")
#
#
def question_3b():
    df = pd.read_csv("./Data/Istanbul.csv", header=0)
    X = df.drop(["date", "EM"], axis=1).values
    Y = df["EM"].values
    # Time comparison
    time_values_dt = []
    time_values_rt = []
    size_values = np.arange(1, X.shape[0], 30, dtype=np.uint64)
    for size_value in size_values:
        trainX = X[:size_value, :]
        trainY = Y[:size_value]
        # Measure DTLearner
        dt_learner = dt.DTLearner(leaf_size=1)
        start = time.time()
        dt_learner.addEvidence(trainX, trainY)
        end = time.time()
        time_values_dt.append(end-start)

        # Measure RTLearner
        start = time.time()
        rt_learner = rt.RTLearner(leaf_size=1)
        rt_learner.addEvidence(trainX, trainY)
        end = time.time()
        time_values_rt.append(end-start)

    fig, ax = plt.subplots()
    pd.DataFrame({

        "DTLearner training time": time_values_dt,
        "RTLearner training time": time_values_rt
    }, index=size_values).plot(
        ax=ax,
        style="o-",
        title="Training time comparison on istanbul data"
    )
    plt.xlabel("Size of training set")
    plt.ylabel("Time (seconds)")
    ax.xaxis.grid(True, which='major', linestyle='dotted')
    ax.yaxis.grid(True, which='major', linestyle='dotted')
    plt.legend(loc=2)
    plt.tight_layout()
    #plt.show()
    plt.savefig("Q3b.png")
#
#

if __name__=="__main__":

    #test()

    print ("Q1...", question_1(), "done.")

    print ("Q2a...", question_2a(),  "done.")

    print("Q3a...", question_3a(),  "done.") # RMSE
    #
    print("Q3b...", question_3b(),  "done.") # Training times
    #

