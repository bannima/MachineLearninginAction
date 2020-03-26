###KNN工程实现


###FAQ
1.KD树的查询实现？如何在kd树中查找k个最近领点？

第一个问题，详见实现tree.py的_nearest_k_point函数。
总体而言是深度优先遍历树加上迭代方法实现。

第二个问题，通过一个包含k个元素的堆来实现，具体使用heapq，注意Node元素此时需要继承__lt__函数来避免出现错误。

2.kd树和球树区别

3.如何选择切分维度，维度是否可以重复？

最简单的方法是循环各个维度，即维度axis=（axis+1）%m，m表示数据的特征数目。这种方法优点在于实现较为简单，
缺点也很明显就是无法从将数据波动性较大的维度选择出来，建立的树存在过深的情况。

一种改进方法在于选择数据波动性较大的特征维度，在计算角度是方差variance较大的特征，在计算时，给定数据集，
先计算各个维度的方法，在从其中选择较大的维度作为当前数据的切分维度。

这其中一个问题在于，如果上一层的数据集切分维度和下一层的数据集切分维度相同是否有影响？

这个问题要从kd tree的回溯过程来理解，kd tree的重要意义在于将数据按照维度切分，从而将点限制在局部的范围，即距离
相等的点大致在一个范围。在回溯的过程中，不同范围内的点，通过计算筛选点到父节点的切分维度的垂直距离，来作为筛
选点到超平面的距离，若点到超平面距离小于当前最有距离，则有可能存在更近点，反之则一定不存在更近点（通过距离的
公式可得到这一结论）。回到最初的问题，答案应该是没有影响，父类节点将两个超空间进行了切分，而子节点如何切分对于
点到父类点确定的超平面距离并没有影响，因此维度的选择在不同层次之间是相互独立的。

4.如何选择找到当前最近点？

理想情况下，即平衡的满二叉树情况下这个问题无需求解，但是问题在于可能存在内部节点只有单个分支，即只有左子树
或右子树的情况，若当前父节点的维度需要分到空的分支时，这里可以用存在的分支来近似代替不存在的分支，因为计算上
都要进行回溯。

5.如何回溯查找kd树？

kd树的回溯查找直观上比较简单，但需要处理好一些细节问题，如如何避免兄弟节点重复查询的死循环，如果当前节点只有
一个分支节点，而要比较的节点恰巧往另一不存在的分支如何处理？

先回答第二个问题，具体做法很简单，就是只有单个节点的情况，直接分到那个分支，因为回溯的过程中都会进行查找，不影响结果。
而且kd树的左右子树空间都在回溯的过程中会被搜索到。

再回答第一个问题，需要保存好路径，具体在每个节点的循环迭代中，先从根节点到叶子结点的匹配过程保存好，如果回退到跟节点
时候，需要判断若当前节点是跟节点，就不要进行兄弟节点的查找，避免回退到之前查询过的兄弟节点，具体实现时，只需要进行
判断（具体见tree.py的_nearest_k_point函数的if cur_node.brother and (len(path)!=0)判断条件），len(path)==0
表示当前节点为此次迭代的根节点，跳过进行兄弟节点的判断，避免重复计算。

6.heapq对于Node的要求。

在python3中需要实现__lt__函数，避免出现错误unorderable types: Node() < Node()，
可参考：
https://stackoverflow.com/questions/34005451/typeerror-unorderable-types-dict-dict-in-heapq
https://www.jb51.net/article/85716.htm



###参考资料
1.An intoductory tutorial on kd-trees, CMU, Andrew W. Moore,1991
2.《统计学习方法》李航 第一版
3. https://github.com/tushushu/imylu/blob/master/imylu/utils/kd_tree.py
4.https://stackoverflow.com/questions/34005451/typeerror-unorderable-types-dict-dict-in-heapq
5.https://www.jb51.net/article/85716.htm
6.https://docs.python.org/zh-cn/3.6/library/heapq.html
7.https://blog.csdn.net/Yan456jie/article/details/52074141
8.https://leileiluoluo.com/posts/kdtree-algorithm-and-implementation.html