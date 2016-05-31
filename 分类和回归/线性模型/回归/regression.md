# 线性回归

&emsp;&emsp;回归问题的条件或者说前提是
- 1） 收集的数据
- 2） 假设的模型，即一个函数，这个函数里含有未知的参数，通过学习，可以估计出参数。然后利用这个模型去预测/分类新的数据。

## 1 线性回归的概念

&emsp;&emsp;线性回归假设特征和结果都满足线性。即不大于一次方。收集的数据中，每一个分量，就可以看做一个特征数据。每个特征至少对应一个未知的参数。这样就形成了一个线性模型函数，向量表示形式：

<div  align="center"><img src="imgs/1.1.png" width = "120" height = "30" alt="1.1" align="center" /></div>

&emsp;&emsp;这个就是一个组合问题，已知一些数据，如何求里面的未知参数，给出一个最优解。 一个线性矩阵方程，直接求解，很可能无法直接求解。有唯一解的数据集，微乎其微。

&emsp;&emsp;基本上都是解不存在的超定方程组。因此，需要退一步，将参数求解问题，转化为求最小误差问题，求出一个最接近的解，这就是一个松弛求解。

&emsp;&emsp;在回归问题中，线性最小二乘是最普遍的求最小误差的形式。它的损失函数就是二乘损失。如下公式**（1）**所示：

<div  align="center"><img src="imgs/1.2.png" width = "240" height = "50" alt="1.2" align="center" /></div>

&emsp;&emsp;根据使用的正则化类型的不同，回归算法也会有不同。普通最小二乘和线性最小二乘回归不使用正则化方法。`ridge`回归使用`L2`正则化，`lasso`回归使用`L1`正则化。

## 2 线性回归源码分析

### 2.1 实例

```scala
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.linalg.Vectors
// 获取数据
val data = sc.textFile("data/mllib/ridge-data/lpsa.data")
val parsedData = data.map { line =>
  val parts = line.split(',')
  LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
}.cache()
//训练模型
val numIterations = 100
val stepSize = 0.00000001
val model = LinearRegressionWithSGD.train(parsedData, numIterations, stepSize)
// 评价
val valuesAndPreds = parsedData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val MSE = valuesAndPreds.map{case(v, p) => math.pow((v - p), 2)}.mean()
println("training Mean Squared Error = " + MSE)
```

### 2.2 代码实现

&emsp;&emsp;和逻辑回归一样，训练过程均使用`GeneralizedLinearModel`中的`run`训练，只是训练使用的`Gradient`和`Updater`不同。在一般的线性回归中，使用`LeastSquaresGradient`计算梯度，使用`SimpleUpdater`进行更新。
它的实现过程分为4步。参加[逻辑回归](../逻辑回归/logic-regression.md)了解这五步的详细情况。我们只需要了解`LeastSquaresGradient`和`SimpleUpdater`的实现。

&emsp;&emsp;普通线性回归的损失函数是最小二乘损失，如上面的公式**（1）**所示。

```scala
class LeastSquaresGradient extends Gradient {
  override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    //diff = xw-y
    val diff = dot(data, weights) - label
    val loss = diff * diff / 2.0
    val gradient = data.copy
    //gradient = diff * gradient
    scal(diff, gradient)
    (gradient, loss)
  }
  override def compute(
      data: Vector,
      label: Double,
      weights: Vector,
      cumGradient: Vector): Double = {
    //diff = xw-y
    val diff = dot(data, weights) - label
    //计算梯度
    //cumGradient += diff * data
    axpy(diff, data, cumGradient)
    diff * diff / 2.0
  }
}
```
&emsp;&emsp;普通线性回归的不适用正则化方法，所以它用`SimpleUpdater`实现`Updater`。

```scala
class SimpleUpdater extends Updater {
  override def compute(
      weightsOld: Vector,
      gradient: Vector,
      stepSize: Double,
      iter: Int,
      regParam: Double): (Vector, Double) = {
    val thisIterStepSize = stepSize / math.sqrt(iter)
    val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
    //计算 y += x * a，即 brzWeights -= thisIterStepSize * gradient.toBreeze
    //梯度下降方向
    brzAxpy(-thisIterStepSize, gradient.toBreeze, brzWeights)
    (Vectors.fromBreeze(brzWeights), 0)
  }
}
```
&emsp;&emsp;这里`thisIterStepSize`表示参数沿负梯度方向改变的速率，它随着迭代次数的增多而减小。

## 3 Lasso回归源码分析

&emsp;&emsp;`lasso`回归和普通线性回归不同的地方是，它使用`L1`正则化方法。即使用`L1Updater`。

```scala
class L1Updater extends Updater {
  override def compute(
      weightsOld: Vector,
      gradient: Vector,
      stepSize: Double,
      iter: Int,
      regParam: Double): (Vector, Double) = {
    val thisIterStepSize = stepSize / math.sqrt(iter)
    //计算 y += x * a，即 brzWeights -= thisIterStepSize * gradient.toBreeze
    //梯度下降方向
    val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
    brzAxpy(-thisIterStepSize, gradient.toBreeze, brzWeights)
    // Apply proximal operator (soft thresholding)
    val shrinkageVal = regParam * thisIterStepSize
    var i = 0
    val len = brzWeights.length
    while (i < len) {
      val wi = brzWeights(i)
      brzWeights(i) = signum(wi) * max(0.0, abs(wi) - shrinkageVal)
      i += 1
    }
    (Vectors.fromBreeze(brzWeights), brzNorm(brzWeights, 1.0) * regParam)
  }
}
```
&emsp;&emsp;这个类解决`L1`范式正则化问题。这里`thisIterStepSize`表示参数沿负梯度方向改变的速率，它随着迭代次数的增多而减小。该实现没有使用[线性模型](../readme.md)中介绍的子梯度方法，而是使用了邻近算子（`proximal operator`）来解决，该方法的结果拥有更好的稀疏性。
`L1`范式的邻近算子是软阈值（`soft-thresholding`）函数。

- 当`w >  shrinkageVal`时，权重组件等于`w-shrinkageVal`

- 当`w < -shrinkageVal`时，权重组件等于`w+shrinkageVal`

- 当`-shrinkageVal < w < shrinkageVal`时，权重组件等于0

&emsp;&emsp;`signum`函数是子梯度函数，当`w<0`时，返回-1，当`w>0`时，返回1，当`w=0`时，返回0。

## 4 ridge回归源码分析

&emsp;&emsp;`ridge`回归的训练过程也是一样，它使用`L2`正则化方法。

```scala
class SquaredL2Updater extends Updater {
  override def compute(
      weightsOld: Vector,
      gradient: Vector,
      stepSize: Double,
      iter: Int,
      regParam: Double): (Vector, Double) = {
    // w' = w - thisIterStepSize * (gradient + regParam * w)
    // w' = (1 - thisIterStepSize * regParam) * w - thisIterStepSize * gradient
    //表示步长，即负梯度方向的大小
    val thisIterStepSize = stepSize / math.sqrt(iter)
    val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
    //正则化，brzWeights每行数据均乘以(1.0 - thisIterStepSize * regParam)
    brzWeights :*= (1.0 - thisIterStepSize * regParam)
    //y += x * a，即brzWeights -= gradient * thisInterStepSize
    brzAxpy(-thisIterStepSize, gradient.toBreeze, brzWeights)
    //正则化||w||_2
    val norm = brzNorm(brzWeights, 2.0)
    (Vectors.fromBreeze(brzWeights), 0.5 * regParam * norm * norm)
  }
}
```
&emsp;&emsp;该函数的实现规则是：

```scala
 w1 = w - thisIterStepSize * (gradient + regParam * w)
 w1 = (1 - thisIterStepSize * regParam) * w - thisIterStepSize * gradient
```
&emsp;&emsp;这里`thisIterStepSize`表示参数沿负梯度方向改变的速率（即步长），它随着迭代次数的增多而减小。
