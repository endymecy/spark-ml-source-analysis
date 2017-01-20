# 带权最小二乘

&emsp;&emsp;给定n个带权的观察样本$(w_i,a_i,b_i)$:

- $w_i$表示第i个观察样本的权重；
- $a_i$表示第i个观察样本的特征向量；
- $b_i$表示第i个观察样本的标签。

&emsp;&emsp;每个观察样本的特征数是m。我们使用下面的带权最小二乘公式作为目标函数：

$$minimize_{x}\frac{1}{2} \sum_{i=1}^n \frac{w_i(a_i^T x -b_i)^2}{\sum_{k=1}^n w_k} + \frac{1}{2}\frac{\lambda}{\delta}\sum_{j=1}^m(\sigma_{j} x_{j})^2$$

&emsp;&emsp;其中$\lambda$是正则化参数，$\delta$是标签的总体标准差，$\sigma_j$是第j个特征列的总体标准差。

&emsp;&emsp;这个目标函数有一个解析解法，它仅仅需要一次处理样本来搜集必要的统计数据去求解。与原始数据集必须存储在分布式系统上不同，
如果特征数相对较小，这些统计数据可以加载进单机的内存中，然后在`driver`端使用乔里斯基分解求解目标函数。

&emsp;&emsp;`spark ml`中使用`WeightedLeastSquares`求解带权最小二乘问题。`WeightedLeastSquares`仅仅支持`L2`正则化，并且提供了正则化和标准化
的开关。为了使正太方程（`normal equation`）方法有效，特征数不能超过4096。如果超过4096，用`L-BFGS`代替。下面从代码层面介绍带权最小二乘优化算法
的实现。

## 代码解析

&emsp;&emsp;我们首先看看`WeightedLeastSquares`的参数及其含义。

```scala
private[ml] class WeightedLeastSquares(
    val fitIntercept: Boolean,  //是否使用截距
    val regParam: Double,    //L2正则化参数，指上面公式中的lambda
    val elasticNetParam: Double,  // alpha
    val standardizeFeatures: Boolean, // 是否标准化特征
    val standardizeLabel: Boolean,  // 是否标准化标签
    val solverType: WeightedLeastSquares.Solver = WeightedLeastSquares.Auto,
    val maxIter: Int = 100, // 迭代次数
    val tol: Double = 1e-6) extends Logging with Serializable 
    
 sealed trait Solver
 case object Auto extends Solver
 case object Cholesky extends Solver
 case object QuasiNewton extends Solver
```
&emsp;&emsp;在上面的代码中，`standardizeFeatures`决定是否标准化特征，如果为真，则$\sigma_j$是A第j个特征列的总体标准差，否则$\sigma_j$为1。
`standardizeLabel`决定是否标准化标签，如果为真，则$\delta$是标签b的总体标准差，否则$\delta$为1。`solverType`指定求解的类型，有`Auto`，`Cholesky`
和`QuasiNewton`三种选择。`tol`表示迭代的收敛阈值，仅仅在`solverType`为`QuasiNewton`时可用。


