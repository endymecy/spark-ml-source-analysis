# 梯度提升树

&emsp;&emsp;梯度提升（`gradient boosting`）算法的核心在于，每棵树是从先前所有树的残差中来学习。利用的是当前模型中损失函数的负梯度值作为提升树算法中的残差的近似值，进而拟合一棵回归（分类）树。

&emsp;&emsp;梯度提升属于`Boost`算法的一种，也可以说是`Boost`算法的一种改进，原始的`Boost`算法是在算法开始时，为每一个样本赋上一个相等的权重值，也就是说，最开始的时候，大家都是一样重要的。
在每一次训练中得到的模型，会使得数据点的估计有所差异，所以在每一步结束后，我们需要对权重值进行处理，而处理的方式就是通过增加错分类点的权重，同时减少错分类点的权重，这样使得某些点如果老是被分错，那么就会被“严重关注”，也就被赋上一个很高的权重。
然后等进行了`N`次迭代，将会得到`N`个简单的基分类器（`basic learner`），最后将它们组合起来，可以对它们进行加权（错误率越大的基分类器权重值越小，错误率越小的基分类器权重值越大）、或者让它们进行投票等得到一个最终的模型。

&emsp;&emsp;`Gradient Boost`与传统的`Boost`有着很大的区别，它的每一次计算都是为了减少上一次的残差(`residual`)，而为了减少这些残差，可以在残差减少的梯度(`Gradient`)方向上建立一个新模型。所以说，在`Gradient Boost`中，每个新模型的建立是为了使得先前模型残差往梯度方向减少，
与传统的`Boost`算法对正确、错误的样本进行加权有着极大的区别。

## 1 梯度提升

&emsp;&emsp;根据参考文献【1】的介绍，梯度提升算法的算法流程如下所示：

<div  align="center"><img src="imgs/1.1.png" width = "800" height = "350" alt="1.1" align="center" /></div>

&emsp;&emsp;在上述的流程中，`psi`表示的是损失函数。在`MLlib`中，提供的损失函数有三种。如下图所示。

<div  align="center"><img src="imgs/1.3.png" width = "800" height = "190" alt="1.3" align="center" /></div>

&emsp;&emsp;第一个对数损失用于分类，后两个平方误差和绝对误差用于回归。

## 2 随机梯度提升

&emsp;&emsp;有文献证明，注入随机性到上述的过程中可以提高函数估计的性能。受到`Breiman`的影响，将随机性作为一个考虑的因素。在每次迭代中，随机的在训练集中抽取一个子样本集，然后在后续的操作中用这个子样本集代替全体样本。
这就形成了随机梯度提升算法。它的流程如下所示：

<div  align="center"><img src="imgs/1.2.png" width = "800" height = "350" alt="1.2" align="center" /></div>

## 3 实例和源码分析

### 3.1 实例

&emsp;&emsp;下面的代码是分类的例子。

```scala
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.mllib.util.MLUtils
// 准备数据
val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
// Split the data into training and test sets (30% held out for testing)
val splits = data.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))
// 训练模型
// The defaultParams for Classification use LogLoss by default.
val boostingStrategy = BoostingStrategy.defaultParams("Classification")
boostingStrategy.numIterations = 3 // Note: Use more iterations in practice.
boostingStrategy.treeStrategy.numClasses = 2
boostingStrategy.treeStrategy.maxDepth = 5
// Empty categoricalFeaturesInfo indicates all features are continuous.
boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()
val model = GradientBoostedTrees.train(trainingData, boostingStrategy)
// 用测试数据评价模型
val labelAndPreds = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
println("Test Error = " + testErr)
println("Learned classification GBT model:\n" + model.toDebugString)
```
&emsp;&emsp;下面的代码是回归的例子。

```scala
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.mllib.util.MLUtils
// 准备数据
val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
// Split the data into training and test sets (30% held out for testing)
val splits = data.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))
// 训练模型
// The defaultParams for Regression use SquaredError by default.
val boostingStrategy = BoostingStrategy.defaultParams("Regression")
boostingStrategy.numIterations = 3 // Note: Use more iterations in practice.
boostingStrategy.treeStrategy.maxDepth = 5
// Empty categoricalFeaturesInfo indicates all features are continuous.
boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()
val model = GradientBoostedTrees.train(trainingData, boostingStrategy)
// 用测试数据评价模型
val labelsAndPredictions = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val testMSE = labelsAndPredictions.map{ case(v, p) => math.pow((v - p), 2)}.mean()
println("Test Mean Squared Error = " + testMSE)
println("Learned regression GBT model:\n" + model.toDebugString)
```

# 参考文献

【1】[Stochastic Gradient Boost](https://statweb.stanford.edu/~jhf/ftp/stobst.pdf)

【2】[机器学习算法-梯度树提升GTB（GBRT）](http://www.07net01.com/2015/08/918187.html)
