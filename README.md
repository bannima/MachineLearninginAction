# Machine Learning Models in Action

## 机器学习实践专题


主要根据李航老师《统计学习方法》，仔细梳理主流机器学习模型，编码实现。
具体涵盖以下方法：

### 第1章 统计学习方法概论

### 第2章 [感知机](https://github.com/bannima/MachineLearninginAction/blob/master/linear_model/perceptron.py)

#### 2.1 [Logistic Regression](https://github.com/bannima/MachineLearninginAction/blob/master/linear_model/perceptron.py)

#### 2.1 [Linear Regression]()


### 第3章 [k近邻法](https://github.com/bannima/MachineLearninginAction/tree/master/neighbors)

K Nearest Neighbor with KD Tree

### 第4章 [朴素贝叶斯](https://github.com/bannima/MachineLearninginAction/tree/master/bayes)

朴素贝叶斯

### 第5章 [决策树](https://github.com/bannima/MachineLearninginAction/tree/master/tree)

#### 5.1 [ID3](https://github.com/bannima/MachineLearninginAction/blob/master/tree/tree.py)

基于信息增益作为生成决策树的准则。

#### 5.2 C4.5

基于信息增益比作为生成决策树的准则。

#### 5.3 [CART分类树和CART回归树](https://github.com/bannima/MachineLearninginAction/blob/master/tree/tree.py)

分类与回归树，基于二叉树，可分类也可回归。

### 第6章 [逻辑斯蒂回归与最大熵模型]()

#### 6.1  [逻辑斯蒂回归](https://github.com/bannima/MachineLearninginAction/blob/master/linear_model/regression.py)

#### 6.2 [最大熵模型](https://github.com/bannima/MachineLearninginAction/blob/master/linear_model/maximum_entropy.py)

### 第7章 [支持向量机](https://github.com/bannima/MachineLearninginAction/tree/master/svm)

Support Vector Machines using SMO

### 第8章 [提升方法](https://github.com/bannima/MachineLearninginAction/tree/master/ensemble)

#### 8.1 [Adaboost](https://github.com/bannima/MachineLearninginAction/blob/master/ensemble/boosting.py)

boosting家族中具有代表性的方法一种，基于前向加法模型，基本学习器为基本分类器，只能处理二分类问题，可看作GBDT的特例，
此时基础学习器为基本分类器，损失函数为指数函数。

#### 8.2 [Random Forest](https://github.com/bannima/MachineLearninginAction/blob/master/ensemble/bagging.py)

Bagging的代表性算法，基于样本随机采样(行采样)和部分特征采样（列采样），根据基础决策树不同，可分别用于回归或者二分类、多分类问题。


#### 8.3 [GBDT和GBRT](https://github.com/bannima/MachineLearninginAction/blob/master/ensemble/gradient_boosting.py)
梯度提升分类树和梯度提升回归树。
基于前向加法模型，基本学习器采用决策树，可解决分类以及回归问题，加法模型每一步都在拟合损失函数的负梯度。


### 第9章 EM算法及其推广

#### 9.1 Gaussian misture model,高斯混合模型


### 第10章 [隐马尔可夫模型](https://github.com/bannima/MachineLearninginAction/tree/master/hmm)

Hidden Markov Model,隐马尔可夫模型


### 第11章 [条件随机场](https://github.com/bannima/MachineLearninginAction/tree/master/crf)

Conditional Random Field 条件随机场，利用梯度下降学习参数，用维特比算法进行序列预测，
并用中文分词实验中检验结果。

### 第12章 统计学习方法总结

### 第13章 无监督学习概论

### 第14章 聚类方法

#### 14.1 K-Means

#### 14.2 Hierarchical Cluster

#### 14.33 DBSCAN


### 第15章 奇异值分解


### 第16章 主成分分析

### 第17章 潜在语义分析
#### 17.1 LSA

### 第18章 概率潜在语义分析

#### 18.1 PLSA


### 第19章 [马尔可夫链蒙特卡罗法](https://github.com/bannima/MachineLearninginAction/tree/master/sampling/test)

#### 19.1 Metropolis-Hasting

#### 19.2 Gibbs

### 第20章 潜在狄利克雷分配

#### 20.1 LDA

### 第21章 [PageRank算法](https://github.com/bannima/MachineLearninginAction/blob/master/pagerank/page_rank.py)


## More

###  [1.神经网络](https://github.com/bannima/MachineLearninginAction/tree/master/neural_networks)
#### 1 [DNN](https://github.com/bannima/MachineLearninginAction/blob/master/neural_networks/dnn.py)

#### 2 CNN(coming soon)

#### 3 RNN

#### 3.1 LSTM(coming soon)

### [2.Embedding]()

#### 1 Word2Vec


### 参考资料

1 《统计学习方法》李航 第二版

2 《机器学习实战》

3 《机器学习》周志华

