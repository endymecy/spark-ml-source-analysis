# 二分`k-means`算法

&emsp;&emsp;二分`k-means`算法是分层聚类（[Hierarchical clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering)）的一种，分层聚类是聚类分析中常用的方法。
分层聚类的策略一般有两种：

- 聚合。这是一种`自底向上`的方法，每一个观察者以自己为一类，然后两两结合

- 分裂。这是一种`自顶向下`的方法，所有观察者同为一类，然后递归地分裂它们

&emsp;&emsp;二分`k-means`算法使用分裂法。

## 二分`k-means`的步骤

&emsp;&emsp;二分`k-means`算法是`k-means`算法的改进算法，相比`k-means`算法，它有如下优点：

- 二分`k-means`算法可以加速`k-means`算法的执行速度，因为它的相似度计算少了

- 能够克服`k-means`收敛于局部最小的缺点

&emsp;&emsp;二分`k-means`算法的一般流程如下所示：

- （1）把所有数据初始化为一个簇，将这个簇分为两个簇

- （2）选择能最大程度降低聚类代价函数（也就是误差平方和`SSE`）的簇划分为两个簇

- （3）一直重复（2）步，直到满足我们给定的聚类数

