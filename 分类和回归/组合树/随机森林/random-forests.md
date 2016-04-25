# 随机森林

## 1 集成学习

&emsp;&emsp;集成学习通过构建并结合多个学习器来完成学习任务，有时也被称为多分类器系统。集成学习通过将多个学习器进行结合，常可获得比单一学习器显著优越的泛化能力。

&emsp;&emsp;根据个体学习器的生成方式，目前的集成学习方法大致可以分为两大类。即个体学习器之间存在强依赖性，必须串行生成的序列化方法以及个体学习器之间不存在强依赖性，可同时生成的并行化方法。
前者的代表是`Boosting`，后者的代表是`Bagging`和随机森林。下面详细介绍`Bagging`和随机森林

## 2 Bagging

&emsp;&emsp;`Bagging`采用自助采样法(`bootstrap sampling`)采样数据。给定包含`m`个样本的数据集，我们先随机取出一个样本放入采样集中，再把该样本放入初始数据集，使得下次采样时，样本仍可能被选中，
这样，经过`m`次随机采样操作，我们得到汗`m`个样本的采样集。

&emsp;&emsp;按照此方式，我们可以采样出`T`个含`m`个训练样本的采样集，然后基于每个采样集训练出一个基本学习器，再将这些基本学习器进行结合。这就是`Bagging`的一般流程。在对预测输出进行结合时，`Bagging`通常使用简单投票法，
对回归问题使用简单平均法。若分类预测时，出现两个类收到同样票数的情形，则最简单的做法是随机选择一个，也可以进一步考察学习器投票的置信度来确定最终胜者。

&emsp;&emsp;`Bagging`的算法描述如下图所示。

<div  align="center"><img src="imgs/1.1.png" width = "400" height = "220" alt="1.1" align="center" /></div>

## 3 随机森林

&emsp;&emsp;随机森林是`Bagging`的一个扩展变体。随机森林在以决策树为基学习器构建`Bagging`集成的基础上，进一步在决策树的训练过程中引入了随机属性选择。具体来讲，传统决策树在选择划分属性时，
在当前节点的属性集合（假设有`d`个属性）中选择一个最优属性；而在随机森林中，对基决策树的每个节点，先从该节点的属性集合中随机选择一个包含`k`个属性的子集，然后再从这个子集中选择一个最优属性用于划分。
这里的参数`k`控制了随机性的引入程度。若令`k=d`，则基决策树的构建与传统决策树相同；若令`k=1`，则是随机选择一个属性用于划分。在`MLlib`中，有两种选择用于分类，即`k=log2(d)`、`k=sqrt(d)`；
一种选择用于回归，即`k=1/3d`。在源码分析中会详细介绍。

&emsp;&emsp;可以看出，随机森林对`Bagging`只做了小改动，但是与`Bagging`中基学习器的“多样性”仅仅通过样本扰动（通过对初始训练集采样）而来不同，随机森林中基学习器的多样性不仅来自样本扰动，还来自属性扰动。
这使得最终集成的泛化性能可通过个体学习器之间差异度的增加而进一步提升。

## 4 使用实例

&emsp;&emsp;下面的例子用于分类。

```scala
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
// Load and parse the data file.
val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
// Split the data into training and test sets (30% held out for testing)
val splits = data.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))
// Train a RandomForest model.
// Empty categoricalFeaturesInfo indicates all features are continuous.
val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val numTrees = 3 // Use more in practice.
val featureSubsetStrategy = "auto" // Let the algorithm choose.
val impurity = "gini"
val maxDepth = 4
val maxBins = 32
val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
  numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
// Evaluate model on test instances and compute test error
val labelAndPreds = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
println("Test Error = " + testErr)
println("Learned classification forest model:\n" + model.toDebugString)
```

&emsp;&emsp;下面的例子用于回归。

```scala
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
// Load and parse the data file.
val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
// Split the data into training and test sets (30% held out for testing)
val splits = data.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))
// Train a RandomForest model.
// Empty categoricalFeaturesInfo indicates all features are continuous.
val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val numTrees = 3 // Use more in practice.
val featureSubsetStrategy = "auto" // Let the algorithm choose.
val impurity = "variance"
val maxDepth = 4
val maxBins = 32
val model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo,
  numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
// Evaluate model on test instances and compute test error
val labelsAndPredictions = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val testMSE = labelsAndPredictions.map{ case(v, p) => math.pow((v - p), 2)}.mean()
println("Test Mean Squared Error = " + testMSE)
println("Learned regression forest model:\n" + model.toDebugString)
```