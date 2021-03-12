# Machine Learning Models in Action

## 机器学习实践专题


主要根据李航老师《统计学习方法》，仔细梳理主流机器学习模型，编码实现。
具体涵盖以下方法：


### 1 [linear_model](https://github.com/bannima/MachineLearninginAction/tree/master/linear_model)

1.1 [Perceptron](https://github.com/bannima/MachineLearninginAction/blob/master/linear_model/perceptron.py)

1.2 [Linear Regression]()

1.3 [Logistic Regression](https://github.com/bannima/MachineLearninginAction/blob/master/linear_model/regression.py)

1.4 [Maximum Entropy Model](https://github.com/bannima/MachineLearninginAction/blob/master/linear_model/maximum_entropy.py)

### 2 [neighbors](https://github.com/bannima/MachineLearninginAction/tree/master/neighbors)

K Nearest Neighbor with KD Tree

### 3 [bayes](https://github.com/bannima/MachineLearninginAction/tree/master/bayes)

朴素贝叶斯

### 4 [tree](https://github.com/bannima/MachineLearninginAction/tree/master/tree)

#### 4.1 [ID3](https://github.com/bannima/MachineLearninginAction/blob/master/tree/tree.py)

基于信息增益作为生成决策树的准则。

#### 4.2 C4.5

基于信息增益比作为生成决策树的准则。

#### 4.3 [CART Classification and Regression](https://github.com/bannima/MachineLearninginAction/blob/master/tree/tree.py)

分类与回归树，基于二叉树，可分类也可回归。

### 5 [Maximum Entropy](https://github.com/bannima/MachineLearninginAction/blob/master/linear_model/maximum_entropy.py)

最大熵模型

### 6 [SVM](https://github.com/bannima/MachineLearninginAction/tree/master/svm)

Support Vector Machines using SMO

### 7 [ensemble](https://github.com/bannima/MachineLearninginAction/tree/master/ensemble)

#### 7.1 [Adaboost](https://github.com/bannima/MachineLearninginAction/blob/master/ensemble/boosting.py)

boosting家族中具有代表性的方法一种，基于前向加法模型，基本学习器为基本分类器，只能处理二分类问题，可看作GBDT的特例，
此时基础学习器为基本分类器，损失函数为指数函数。

#### 7.2 [Random Forest](https://github.com/bannima/MachineLearninginAction/blob/master/ensemble/bagging.py)

Bagging的代表性算法，基于样本随机采样(行采样)和部分特征采样（列采样），根据基础决策树不同，可分别用于回归或者二分类、多分类问题。


#### 7.3 [GBDT](https://github.com/bannima/MachineLearninginAction/blob/master/ensemble/gradient_boosting.py)

基于前向加法模型，基本学习器采用决策树，可解决分类以及回归问题，加法模型每一步都在拟合损失函数的负梯度。


### 8 EM（coming soon）

8.1 Gaussian misture model,高斯混合模型


### 9 [HMM](https://github.com/bannima/MachineLearninginAction/tree/master/hmm)

Hidden Markov Model,隐马尔可夫模型


### 10 [CRF](https://github.com/bannima/MachineLearninginAction/tree/master/crf)

Conditional Random Field 条件随机场，利用梯度下降学习参数，用维特比算法进行序列预测，
并用中文分词实验中检验结果。


### 11 [neural networks](https://github.com/bannima/MachineLearninginAction/tree/master/neural_networks)
1. [DNN](https://github.com/bannima/MachineLearninginAction/blob/master/neural_networks/dnn.py)

2. CNN(coming soon)

3. RNN(coming soon)

3.1 LSTM(coming soon)


### 12 [MCMC](https://github.com/bannima/MachineLearninginAction/tree/master/sampling/test)

1.Metropolis-Hasting

2.Gibbs


### 13 [Topic Model](https://github.com/bannima/MachineLearninginAction/tree/master/topic_modeling)

1.LSA

2.PLSA

3.LDA

### 14 Cluster(Coming soon)

1.K-Means

2.Hierarchical Cluster

3.DBSCAN


### 15 Embedding(Coming soon)

1. Word2Vec



### 参考资料

1 《统计学习方法》李航 第二版

2 《机器学习实战》

3 《机器学习》周志华

