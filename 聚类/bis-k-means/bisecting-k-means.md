# 二分`k-means`算法

&emsp;&emsp;二分`k-means`算法是分层聚类（[Hierarchical clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering)）的一种，分层聚类是聚类分析中常用的方法。
分层聚类的策略一般有两种：

- 聚合。这是一种`自底向上`的方法，每一个观察者初始化本身为一类，然后两两结合
- 分裂。这是一种`自顶向下`的方法，所有观察者初始化为一类，然后递归地分裂它们

&emsp;&emsp;二分`k-means`算法是分裂法的一种。

## 二分`k-means`的步骤

&emsp;&emsp;二分`k-means`算法是`k-means`算法的改进算法，相比`k-means`算法，它有如下优点：

- 二分`k-means`算法可以加速`k-means`算法的执行速度，因为它的相似度计算少了
- 能够克服`k-means`收敛于局部最小的缺点

&emsp;&emsp;二分`k-means`算法的一般流程如下所示：

- （1）把所有数据初始化为一个簇，将这个簇分为两个簇。

- （2）选择能最大程度降低聚类代价函数（也就是误差平方和`SSE`）的簇用`k-means`算法划分为两个簇。误差平方和的公式如下所示，其中<img src="http://www.forkosh.com/mathtex.cgi?{w}_{i}">表示权重值，<img src="http://www.forkosh.com/mathtex.cgi?{y}^{*}">表示该簇所有点的平均值。

<div  align="center"><img src="imgs/dis-k-means.1.1.png" width = "195" height = "60" alt="1.1" align="center" /></div><br />

- （3）一直重复（2）步，直到满足我们给定的聚类数

&emsp;&emsp;以上过程隐含着一个原则是：因为聚类的误差平方和能够衡量聚类性能，该值越小表示数据点越接近于它们的质心，聚类效果就越好。
所以我们就需要对误差平方和最大的簇进行再一次的划分，因为误差平方和越大，表示该簇聚类越不好，越有可能是多个簇被当成一个簇了，所以我们首先需要对这个簇进行划分。

## 二分`k-means`的源码分析

&emsp;&emsp;`spark`在文件`org.apache.spark.mllib.clustering.BisectingKMeans`中实现了二分`k-means`算法。在分步骤分析算法实现之前，我们先来了解`BisectingKMeans`类中参数代表的含义。

```scala
class BisectingKMeans private (
    private var k: Int,
    private var maxIterations: Int,
    private var minDivisibleClusterSize: Double,
    private var seed: Long)
```

&emsp;&emsp;上面代码中，`k`表示叶子簇的期望数，默认情况下为4。如果没有可被切分的叶子簇，实际值会更小。`maxIterations`表示切分簇的`k-means`算法的最大迭代次数，默认为20。
`minDivisibleClusterSize`的值如果大于等于1，它表示一个可切分簇的最小点数量；如果值小于1，它表示可切分簇的点数量占总数的最小比例，该值默认为1。

&emsp;&emsp;`BisectingKMeans`的`run`方法实现了二分`k-means`算法，下面将一步步分析该方法的实现过程。

- （1）初始化数据

&emsp;&emsp;






