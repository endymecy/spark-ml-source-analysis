# 逻辑回归

## 1 逻辑回归模型

&emsp;&emsp;回归是一种很容易理解的模型，就相当于`y=f(x)`，表明自变量`x`与因变量`y`的关系。最常见问题如医生治病时的望、闻、问、切，之后判定病人是否生病或生了什么病，
其中的望、闻、问、切就是获取的自变量`x`，即特征数据，判断是否生病就相当于获取因变量`y`，即预测分类。最简单的回归是[线性回归]()，但是线性回归的鲁棒性很差。

&emsp;&emsp;逻辑回归是一种减小预测范围，将预测值限定为`[0,1]`间的一种回归模型，其回归方程与回归曲线如下图所示。逻辑曲线在`z=0`时，十分敏感，在`z>>0`或`z<<0`处，都不敏感，将预测值限定为`(0,1)`。

<div  align="center"><img src="imgs/1.1.png" width = "590" height = "300" alt="1.1" align="center" /></div><br>

&emsp;&emsp;逻辑回归其实是在线性回归的基础上，套用了一个逻辑函数。上图的`g(z)`就是这个逻辑函数(或称为`Sigmoid`函数)。下面左图是一个线性的决策边界，右图是非线性的决策边界。

<div  align="center"><img src="imgs/1.2.png" width = "900" height = "400" alt="1.2" align="center" /></div><br>

&emsp;&emsp;对于线性边界的情况，边界形式可以归纳为如下公式**(1)**:

<div  align="center"><img src="imgs/1.3.png" width = "370" height = "70" alt="1.3" align="center" /></div><br>

&emsp;&emsp;因此我们可以构造预测函数为如下公式**(2)**:

<div  align="center"><img src="imgs/1.4.png" width = "270" height = "75" alt="1.4" align="center" /></div><br>

&emsp;&emsp;该预测函数表示分类结果为1时的概率。因此对于输入点`x`，分类结果为类别1和类别0的概率分别为如下公式**(3)**：

<div  align="center"><img src="imgs/1.5.png" width = "220" height = "55" alt="1.5" align="center" /></div><br>

&emsp;&emsp;对于训练数据集，特征数据`x={x1, x2, … , xm}`和对应的分类数据`y={y1, y2, … , ym}`。构建逻辑回归模型`f`，最典型的构建方法便是应用极大似然估计。对公式**(3)**取极大似然函数，可以得到如下的公式**(4)**:

<div  align="center"><img src="imgs/1.6.png" width = "500" height = "60" alt="1.6" align="center" /></div><br>

&emsp;&emsp;再对公式**(4)**取对数，可得到公式**(5)**：

<div  align="center"><img src="imgs/1.7.png" width = "590" height = "60" alt="1.7" align="center" /></div><br>

&emsp;&emsp;最大似然估计就是求使`l`取最大值时的`theta`。`MLlib`中提供了两种方法来求这个参数，分别是[梯度下降法](../../最优化算法/梯度下降/gradient-descent.md)和[L-BFGS](../../最优化算法/L-BFGS/lbfgs.md)。

## 2 多分类

&emsp;&emsp;二元逻辑回归可以一般化为[多元逻辑回归](http://en.wikipedia.org/wiki/Multinomial_logistic_regression)用来训练和预测多分类问题。对于多分类问题，算法将会训练出一个多元逻辑回归模型，
它包含`K-1`个二元回归模型。给定一个数据点，`K-1`个模型都会运行，概率最大的类别将会被选为预测类别。

## 3 实例

&emsp;&emsp;下面的例子展示了如何使用逻辑回归。

```scala
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
// 加载训练数据
val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
// 切分数据，training (60%) and test (40%).
val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0).cache()
val test = splits(1)
// 训练模型
val model = new LogisticRegressionWithLBFGS()
  .setNumClasses(10)
  .run(training)
// Compute raw scores on the test set.
val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}
// Get evaluation metrics.
val metrics = new MulticlassMetrics(predictionAndLabels)
val precision = metrics.precision
println("Precision = " + precision)
// 保存和加载模型
model.save(sc, "myModelPath")
val sameModel = LogisticRegressionModel.load(sc, "myModelPath")
```

## 4 源码分析

&emsp;&emsp;如上所述，在`MLlib`中，分别使用了梯度下降法和`L-BFGS`实现逻辑回归。这两个算法的实现我们会在最优化章节介绍，这里我们介绍公共的部分。

&emsp;&emsp;`LogisticRegressionWithLBFGS`和`LogisticRegressionWithSGD`的入口函数均是`GeneralizedLinearAlgorithm.run`，下面详细分析该方法。

```scala
def run(input: RDD[LabeledPoint]): M = {
    if (numFeatures < 0) {
      //计算特征数
      numFeatures = input.map(_.features.size).first()
    }
    val initialWeights = {
          if (numOfLinearPredictor == 1) {
            Vectors.zeros(numFeatures)
          } else if (addIntercept) {
            Vectors.zeros((numFeatures + 1) * numOfLinearPredictor)
          } else {
            Vectors.zeros(numFeatures * numOfLinearPredictor)
          }
    }
    run(input, initialWeights)
}
```
&emsp;&emsp;上面的代码初始化权重向量，向量的值均初始化为0。需要注意的是，`addIntercept`表示是否添加截距(`Intercept`)，默认是不添加的。`numOfLinearPredictor`表示二元逻辑回归模型的个数。
我们重点看`run(input, initialWeights)`的实现。它的实现分四步。

- **1** 根据提供的参数缩放特征并添加`Intercept`

```scala
val scaler = if (useFeatureScaling) {
      new StandardScaler(withStd = true, withMean = false).fit(input.map(_.features))
    } else {
      null
    }
val data =
      if (addIntercept) {
        if (useFeatureScaling) {
          input.map(lp => (lp.label, appendBias(scaler.transform(lp.features)))).cache()
        } else {
          input.map(lp => (lp.label, appendBias(lp.features))).cache()
        }
      } else {
        if (useFeatureScaling) {
          input.map(lp => (lp.label, scaler.transform(lp.features))).cache()
        } else {
          input.map(lp => (lp.label, lp.features))
        }
      }
val initialWeightsWithIntercept = if (addIntercept && numOfLinearPredictor == 1) {
      appendBias(initialWeights)
    } else {
      /** If `numOfLinearPredictor > 1`, initialWeights already contains intercepts. */
      initialWeights
    }
```

&emsp;&emsp;在最优化过程中，收敛速度依赖于训练数据集的条件数(`condition number`)，缩放变量经常可以启发式地减少这些条件数，提高收敛速度。不减少条件数，一些混合有不同范围列的数据集可能不能收敛。
在这里使用`StandardScaler`将数据集的特征进行缩放。详细信息请看[StandardScaler](../../特征抽取和转换/StandardScaler.md)。`appendBias`方法很简单，就是在每个向量后面加一个值为1的项。

```scala
def appendBias(vector: Vector): Vector = {
    vector match {
      case dv: DenseVector =>
        val inputValues = dv.values
        val inputLength = inputValues.length
        val outputValues = Array.ofDim[Double](inputLength + 1)
        System.arraycopy(inputValues, 0, outputValues, 0, inputLength)
        outputValues(inputLength) = 1.0
        Vectors.dense(outputValues)
      case sv: SparseVector =>
        val inputValues = sv.values
        val inputIndices = sv.indices
        val inputValuesLength = inputValues.length
        val dim = sv.size
        val outputValues = Array.ofDim[Double](inputValuesLength + 1)
        val outputIndices = Array.ofDim[Int](inputValuesLength + 1)
        System.arraycopy(inputValues, 0, outputValues, 0, inputValuesLength)
        System.arraycopy(inputIndices, 0, outputIndices, 0, inputValuesLength)
        outputValues(inputValuesLength) = 1.0
        outputIndices(inputValuesLength) = dim
        Vectors.sparse(dim + 1, outputIndices, outputValues)
      case _ => throw new IllegalArgumentException(s"Do not support vector type ${vector.getClass}")
    }
```

- **2** 使用最优化算法计算最终的权重值

```scala
val weightsWithIntercept = optimizer.optimize(data, initialWeightsWithIntercept)
```
&emsp;&emsp;有梯度下降算法和`L-BFGS`两种算法来计算最终的权重值，查看[梯度下降法](../../最优化算法/梯度下降/gradient-descent.md)和[L-BFGS](../../最优化算法/L-BFGS/lbfgs.md)了解详细实现。

- **3** 对最终的权重值进行后处理

```scala
val intercept = if (addIntercept && numOfLinearPredictor == 1) {
      weightsWithIntercept(weightsWithIntercept.size - 1)
    } else {
      0.0
    }
var weights = if (addIntercept && numOfLinearPredictor == 1) {
      Vectors.dense(weightsWithIntercept.toArray.slice(0, weightsWithIntercept.size - 1))
    } else {
      weightsWithIntercept
    }
```
&emsp;&emsp;该端代码获得了`intercept`以及最终的权重值。由于`intercept`和权重是在收缩的空间进行训练的，所以我们需要再把它们转换到元素的空间。数学显示，如果我们仅仅执行标准化而没有减去均值，即`withStd = true, withMean = false`，
那么`intercept`的值并不会发送改变。所以下面的代码仅仅处理权重向量。

```scala
if (useFeatureScaling) {
      if (numOfLinearPredictor == 1) {
        weights = scaler.transform(weights)
      } else {
        var i = 0
        val n = weights.size / numOfLinearPredictor
        val weightsArray = weights.toArray
        while (i < numOfLinearPredictor) {
          //排除intercept
          val start = i * n
          val end = (i + 1) * n - { if (addIntercept) 1 else 0 }
          val partialWeightsArray = scaler.transform(
            Vectors.dense(weightsArray.slice(start, end))).toArray
          System.arraycopy(partialWeightsArray, 0, weightsArray, start, partialWeightsArray.size)
          i += 1
        }
        weights = Vectors.dense(weightsArray)
      }
    }
```

- **4** 创建模型

```scala
createModel(weights, intercept)
```

# 参考文献

【1】[逻辑回归模型(Logistic Regression, LR)基础](http://www.cnblogs.com/sparkwen/p/3441197.html?utm_source=tuicool&utm_medium=referral)

【2】[逻辑回归](http://blog.csdn.net/pakko/article/details/37878837)

