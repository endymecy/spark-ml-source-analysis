# 带权最小二乘

## 1 原理

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

## 2 代码解析

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

### 2.1 训练过程

&emsp;&emsp;`WeightedLeastSquares`接收一个包含（标签，权重，特征）的`RDD`，使用`fit`方法训练，并返回`WeightedLeastSquaresModel`。

```scala
def fit(instances: RDD[Instance]): WeightedLeastSquaresModel
```

&emsp;&emsp;训练过程分为下面几步。

- <b>1 统计样本信息</b>

```scala
val summary = instances.treeAggregate(new Aggregator)(_.add(_), _.merge(_))
```
&emsp;&emsp;使用`treeAggregate`方法来统计样本信息。统计的信息在`Aggregator`类中给出了定义。通过展开上面的目标函数，我们可以知道这些统计信息的含义。

```scala
private class Aggregator extends Serializable {
    var initialized: Boolean = false
    var k: Int = _  // 特征数
    var count: Long = _  // 样本数
    var triK: Int = _  // 对角矩阵保存的元素个数
    var wSum: Double = _  // 权重和
    private var wwSum: Double = _  // 权重的平方和
    private var bSum: Double = _  // 带权标签和
    private var bbSum: Double = _  // 带权标签的平方和
    private var aSum: DenseVector = _  // 带权特征和
    private var abSum: DenseVector = _  // 带权特征标签相乘和
    private var aaSum: DenseVector = _  // 带权特征平方和
    }
```
&emsp;&emsp;方法`add`添加样本的统计信息，方法`merge`合并不同分区的统计信息。代码很简单，如下所示：

```scala
   /**
     * Adds an instance.
     */
    def add(instance: Instance): this.type = {
      val Instance(l, w, f) = instance
      val ak = f.size
      if (!initialized) {
        init(ak)
      }
      assert(ak == k, s"Dimension mismatch. Expect vectors of size $k but got $ak.")
      count += 1L
      wSum += w
      wwSum += w * w
      bSum += w * l
      bbSum += w * l * l
      BLAS.axpy(w, f, aSum)
      BLAS.axpy(w * l, f, abSum)
      BLAS.spr(w, f, aaSum)
      this
    }
    
   /**
     * Merges another [[Aggregator]].
     */
    def merge(other: Aggregator): this.type = {
      if (!other.initialized) {
        this
      } else {
        if (!initialized) {
          init(other.k)
        }
        assert(k == other.k, s"dimension mismatch: this.k = $k but other.k = ${other.k}")
        count += other.count
        wSum += other.wSum
        wwSum += other.wwSum
        bSum += other.bSum
        bbSum += other.bbSum
        BLAS.axpy(1.0, other.aSum, aSum)
        BLAS.axpy(1.0, other.abSum, abSum)
        BLAS.axpy(1.0, other.aaSum, aaSum)
        this
      }
```

&emsp;&emsp;`Aggregator`类给出了以下一些统计信息：

```
aBar: 带权的特征均值
bBar: 带权的标签均值
aaBar: 带权的特征平方均值
bbBar: 带权的标签平方均值
aStd: 带权的特征总体标准差
bStd: 带权的标签总体标准差
aVar: 带权的特征总体方差
```

&emsp;&emsp;计算出这些信息之后，将均值缩放到标准空间，即使每列数据的方差为1。

```scala
// 缩放bBar和 bbBar
val bBar = summary.bBar / bStd
val bbBar = summary.bbBar / (bStd * bStd)

val aStd = summary.aStd
val aStdValues = aStd.values
// 缩放aBar
val aBar = {
      val _aBar = summary.aBar
      val _aBarValues = _aBar.values
      var i = 0
      // scale aBar to standardized space in-place
      while (i < numFeatures) {
        if (aStdValues(i) == 0.0) {
          _aBarValues(i) = 0.0
        } else {
          _aBarValues(i) /= aStdValues(i)
        }
        i += 1
      }
      _aBar
}
val aBarValues = aBar.values
// 缩放 abBar
val abBar = {
      val _abBar = summary.abBar
      val _abBarValues = _abBar.values
      var i = 0
      // scale abBar to standardized space in-place
      while (i < numFeatures) {
        if (aStdValues(i) == 0.0) {
          _abBarValues(i) = 0.0
        } else {
          _abBarValues(i) /= (aStdValues(i) * bStd)
        }
        i += 1
      }
      _abBar
}
val abBarValues = abBar.values
// 缩放aaBar
val aaBar = {
      val _aaBar = summary.aaBar
      val _aaBarValues = _aaBar.values
      var j = 0
      var p = 0
      // scale aaBar to standardized space in-place
      while (j < numFeatures) {
        val aStdJ = aStdValues(j)
        var i = 0
        while (i <= j) {
          val aStdI = aStdValues(i)
          if (aStdJ == 0.0 || aStdI == 0.0) {
            _aaBarValues(p) = 0.0
          } else {
            _aaBarValues(p) /= (aStdI * aStdJ)
          }
          p += 1
          i += 1
        }
        j += 1
      }
      _aaBar
}
val aaBarValues = aaBar.values
```
- <b>2 处理L2正则项</b>



