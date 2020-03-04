##Ensemble Methods
1.bagging

2.boosting

3.stacking

##Random Forest


##Adaboost


##GBDT


FAQ:
1.GBDT如何做分类Classification？
可以将多分类表现转成一个包含各个类别的概率分布，对于每一个类别，
都分别训练一个GBDT回归树，在最后预测时，将各个类别树的结果分别进行预测，
在求概率最大的树作为预测结果。

2.GBDT如何做排序Ranking？


##XGBoost


##参考资料
1.强烈推荐，将GBDT讲述的通俗易懂 http://www.chengli.io/tutorials/gradient_boosting.pdf