##  Bagging and Random Tree


&emsp; In this project, we delve into the Tree-based machine learning model for Finance. Currently, XGBoost or LightBoost, etc. works well for machine learning projects. However, it is more difficult to built them from scracth because of their optimization for memory or parallel/GPU computing. We then start to build Random Forest and Decision Tree from Scracth. Then discuss how leaf size and prune influence on the performance.

#### 1. Overfitting versue leaf size

<p align="center">
  <img src="/img/Q1.png">
</p>
<p align="center">
    <b>
        Fig. 1 RMSE with leaf size for Decision Tree Learner
    </b>
</p>

&emsp;Overfitting does happen in the decision tree learning module. As we know, the overfitting definition is that the training error is small while the testing error is larger, this means the generalization of this algorithm is poor, and cannot be used for future prediction. As shown in Fig 1, when leaf value is less than 10, that will have overfitting.

#### 2. Bagging 

&emsp; We built a **Random Forest** tree by bootstrap aggreating method. In order to discuss the how bootstrap can help solve the overfitting, we choose the bag size as 20, and leaf size differs from 1 ~ 18 to do the analysis.

<p align="center">
  <img src="/img/Q2.png">
  <br>
    <b> 
        Fig. 2 RMSE with leaf size for Bagging Learner
    </b>
</br>
</p>


&emsp; As shown in Fig 2, Bagging can reduce the overfitting, comparing to Fig.1, when leaf size is less than 5, the testing RMSE for BagLearner is less than 0.005. However, the testing RMSE is over 0.006 for Decision Tree as shown in Fig. 1.

&emsp; With the increasing of leaf size, the training RSME is increasing rapidly, however, the testing RMSE is very stable around 0.005~0.0045. As such, Bagging can help reduce overfitting and increase the prediction accuracy, as algorithm has characteristics of randomness.

#### 3. Quantitatively analysis

&emsp; In this chapter, we quantitatively analysis the performance of `Random Tree Learning` and `Decision Tree Learning`. From Fig. 3 and Fig. 4, we can see that Random Tree can not defeat Decision Tree regarding RMSE. Both algorithm will suffer from overfitting, however, random tree algorithm can reduce the overfitting somehow.

<p align="center">
    <title> n</title>
  <img src="/img/3.png" width="425" > 
  <img src="/img/4.png" width="425"/>
  <br> 
        <b>
            Fig. 3 RMSE with leaf size for Bagging Learner
        </b>
    </br>
</p>

&emsp; Howver, in terms of time complexity, random tree is very fast. As Figure 4, decision tree is linear time complexity. In terms of random tree, it is more like a constant time complexity.

<p align="center">
  <img src="/img/5.png" width="600">
  <br>
    <b> 
        Fig. 4 Time complexity for Decision Tree Learning and Random Tree Learning
    </b>
</br>
</p>