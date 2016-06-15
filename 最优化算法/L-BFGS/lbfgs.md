# L-BFGS

# 1 牛顿法

&emsp;&emsp;设`f(x)`是二次可微实函数，又设<img src="http://www.forkosh.com/mathtex.cgi?{x}^{(k)}">是`f(x)`一个极小点的估计，我们把`f(x)`在<img src="http://www.forkosh.com/mathtex.cgi?{x}^{(k)}">展开成`Taylor`级数，
并取二阶近似。

<div  align="center"><img src="imgs/1.1.png" width = "615" height = "45" alt="1.1" align="center" /></div><br>

&emsp;&emsp;上式中最后一项的中间部分表示`f(x)`在<img src="http://www.forkosh.com/mathtex.cgi?{x}^{(k)}">处的`Hesse`矩阵。令上式等于0，可以的到下式：

<div  align="center"><img src="imgs/1.2.png" width = "280" height = "40" alt="1.2" align="center" /></div><br>

&emsp;&emsp;设`Hesse`矩阵可逆，由上式可以得到牛顿法的迭代公式如下**(1.1)**

<div  align="center"><img src="imgs/1.3.png" width = "280" height = "40" alt="1.3" align="center" /></div><br>

&emsp;&emsp;值得注意 ， 当初始点远离极小点时，牛顿法可能不收敛。原因之一是牛顿方向不一定是下降方向，经迭代，目标函数可能上升。此外，即使目标函数下降，得到的点一个不一定沿牛顿方向最好的点或极小点。
因此，我们在牛顿方向上增加一维搜索，提出阻尼牛顿法。其迭代公式是**(1.2)**：

<div  align="center"><img src="imgs/1.4.png" width = "240" height = "60" alt="1.4" align="center" /></div><br>

&emsp;&emsp;其中，`lambda`是由一维搜索（参考文献【1】了解一维搜索）得到的步长，即满足

<div  align="center"><img src="imgs/1.5.png" width = "320" height = "40" alt="1.5" align="center" /></div><br>


# 2 拟牛顿法

## 2.1 拟牛顿条件

&emsp;&emsp;前面介绍了牛顿法，它的突出优点是收敛很快，但是运用牛顿法需要计算二阶偏导数，而且目标函数的`Hesse`矩阵可能非正定。为了克服牛顿法的缺点，人们提出了拟牛顿法，它的基本思想是用不包含二阶导数的矩阵近似牛顿法中的`Hesse`矩阵的逆矩阵。
由于构造近似矩阵的方法不同，因而出现不同的拟牛顿法。

&emsp;&emsp;下面分析怎样构造近似矩阵并用它取代牛顿法中的`Hesse`矩阵的逆。上文**(1.2)**已经给出了牛顿法的迭代公式，为了构造`Hesse`矩阵逆矩阵的近似矩阵<img src="http://www.forkosh.com/mathtex.cgi?{H}_{(k)}">，需要先分析该逆矩阵与一阶导数的关系。

&emsp;&emsp;设在第`k`次迭代之后，得到<img src="http://www.forkosh.com/mathtex.cgi?{x}^{(k+1)}">，我们将目标函数`f(x)`在点<img src="http://www.forkosh.com/mathtex.cgi?{x}^{(k+1)}">展开成`Taylor`级数，
并取二阶近似，得到

<div  align="center"><img src="imgs/2.1.png" width = "630" height = "50" alt="2.1" align="center" /></div><br>

&emsp;&emsp;由此可知，在<img src="http://www.forkosh.com/mathtex.cgi?{x}^{(k+1)}">附近有，

<div  align="center"><img src="imgs/2.2.png" width = "420" height = "60" alt="2.2" align="center" /></div><br>

&emsp;&emsp;记

<div  align="center"><img src="imgs/2.3.png" width = "240" height = "60" alt="2.3" align="center" /></div><br>

&emsp;&emsp;则有

<div  align="center"><img src="imgs/2.4.png" width = "200" height = "30" alt="2.4" align="center" /></div><br>

&emsp;&emsp;又设`Hesse`矩阵可逆，那么上式可以写为如下形式。

<div  align="center"><img src="imgs/2.5.png" width = "215" height = "35" alt="2.5" align="center" /></div><br>

&emsp;&emsp;这样，计算出`p`和`q`之后，就可以通过上面的式子估计`Hesse`矩阵的逆矩阵。因此，为了用不包含二阶导数的矩阵<img src="http://www.forkosh.com/mathtex.cgi?{H}_{(k+1)}">取代牛顿法中`Hesse`矩阵的逆矩阵，有理由令<img src="http://www.forkosh.com/mathtex.cgi?{H}_{(k+1)}">满足公式**(2.1)**：

<div  align="center"><img src="imgs/2.6.png" width = "140" height = "35" alt="2.6" align="center" /></div><br>

&emsp;&emsp;公式**(2.1)**称为拟牛顿条件。

## 2.2 秩1校正

&emsp;&emsp;当`Hesse`矩阵的逆矩阵是对称正定矩阵时，满足拟牛顿条件的矩阵<img src="http://www.forkosh.com/mathtex.cgi?{H}_{(k)}">也应该是对称正定矩阵。构造这样近似矩阵的一般策略是，
<img src="http://www.forkosh.com/mathtex.cgi?{H}_{(1)}">取为任意一个`n`阶对称正定矩阵，通常选择`n`阶单位矩阵`I`，然后通过修正<img src="http://www.forkosh.com/mathtex.cgi?{H}_{(k)}">给定<img src="http://www.forkosh.com/mathtex.cgi?{H}_{(k+1)}">。
令，

<div  align="center"><img src="imgs/2.7.png" width = "150" height = "30" alt="2.7" align="center" /></div><br>

&emsp;&emsp;秩1校正公式写为如下公式**(2.2)**形式。

<div  align="center"><img src="imgs/2.8.png" width = "360" height = "70" alt="2.8" align="center" /></div><br>

## 2.3 DFP算法

&emsp;&emsp;著名的`DFP`方法是`Davidon`首先提出，后来又被`Feltcher`和`Powell`改进的算法，又称为变尺度法。在这种方法中，定义校正矩阵为公式**(2.3)**

<div  align="center"><img src="imgs/2.9.png" width = "280" height = "60" alt="2.9" align="center" /></div><br>

&emsp;&emsp;那么得到的满足拟牛顿条件的`DFP`公式如下**(2.4)**

<div  align="center"><img src="imgs/2.10.png" width = "320" height = "70" alt="2.10" align="center" /></div><br>

&emsp;&emsp;查看文献【1】，了解`DFP`算法的计算步骤。

## 2.4 BFGS算法

&emsp;&emsp;前面利用拟牛顿条件**(2.1)**推导出了`DFP`公式**(2.4)**。下面我们用不含二阶导数的矩阵<img src="http://www.forkosh.com/mathtex.cgi?{B}_{(k+1)}">近似`Hesse`矩阵，从而给出另一种形式的拟牛顿条件**(2.5)**:

<div  align="center"><img src="imgs/2.11.png" width = "140" height = "35" alt="2.11" align="center" /></div><br>

&emsp;&emsp;将公式**(2.1)**的`H`换为`B`，`p`和`q`互换正好可以得到公式**(2.5)**。所以我们可以得到`B`的修正公式**(2.6)**:

<div  align="center"><img src="imgs/2.12.png" width = "320" height = "65" alt="2.12" align="center" /></div><br>

&emsp;&emsp;这个公式称关于矩阵`B`的`BFGS`修正公式，也称为`DFP`公式的对偶公式。设<img src="http://www.forkosh.com/mathtex.cgi?{B}_{(k+1)}">可逆，由公式**(2.1)**以及**(2.5)**可以推出：

<div  align="center"><img src="imgs/2.13.png" width = "110" height = "35" alt="2.13" align="center" /></div><br>

&emsp;&emsp;这样可以得到关于`H`的`BFGS`公式为下面的公式**(2.7)**:

<div  align="center"><img src="imgs/2.14.png" width = "570" height = "60" alt="2.14" align="center" /></div><br>

&emsp;&emsp;这个重要公式是由`Broyden`,`Fletcher`,`Goldfard`和`Shanno`于1970年提出的，所以简称为`BFGS`。数值计算经验表明，它比`DFP`公式还好，因此目前得到广泛应用。

## 2.5 L-BFGS（限制内存BFGS）算法

&emsp;&emsp;在`BFGS`算法中，仍然有缺陷，比如当优化问题规模很大时，矩阵的存储和计算将变得不可行。为了解决这个问题，就有了`L-BFGS`算法。`L-BFGS`即`Limited-memory BFGS`。
`L-BFGS`的基本思想是只保存最近的`m`次迭代信息，从而大大减少数据的存储空间。对照`BFGS`，重新整理一下公式：

<div  align="center"><img src="imgs/2.15.png" width = "200" height = "130" alt="2.15" align="center" /></div><br>

&emsp;&emsp;之前的`BFGS`算法有如下公式**(2.8)**

<div  align="center"><img src="imgs/2.16.png" width = "550" height = "55" alt="2.16" align="center" /></div><br>

&emsp;&emsp;那么同样有

<div  align="center"><img src="imgs/2.17.png" width = "320" height = "30" alt="2.17" align="center" /></div><br>

&emsp;&emsp;将该式子带入到公式**(2.8)**中，可以推导出如下公式

<div  align="center"><img src="imgs/2.18.png" width = "480" height = "150" alt="2.18" align="center" /></div><br>

&emsp;&emsp;假设当前迭代为`k`，只保存最近的`m`次迭代信息，按照上面的方式迭代`m`次，可以得到如下的公式**(2.9)**

<div  align="center"><img src="imgs/2.19.png" width = "500" height = "250" alt="2.19" align="center" /></div><br>

&emsp;&emsp;上面迭代的最终目的就是找到`k`次迭代的可行方向，即

<div  align="center"><img src="imgs/2.20.png" width = "145" height = "30" alt="2.20" align="center" /></div><br>

&emsp;&emsp;为了求可行方向`r`，可以使用`two-loop recursion`算法来求。该算法的计算过程如下，算法中出现的`y`即上文中提到的`t`：

<div  align="center"><img src="imgs/2.21.png" width = "500" height = "350" alt="2.21" align="center" /></div><br>

&emsp;&emsp;算法`L-BFGS`的步骤如下所示。

<div  align="center"><img src="imgs/2.22.png" width = "500" height = "350" alt="2.22" align="center" /></div><br>

## 3 源码解析

### 3.1 实例

```scala
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.{LBFGS, LogisticGradient, SquaredL2Updater}
import org.apache.spark.mllib.util.MLUtils
val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
val numFeatures = data.take(1)(0).features.size
// Split data into training (60%) and test (40%).
val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
// Append 1 into the training data as intercept.
val training = splits(0).map(x => (x.label, MLUtils.appendBias(x.features))).cache()
val test = splits(1)
// Run training algorithm to build the model
val numCorrections = 10
val convergenceTol = 1e-4
val maxNumIterations = 20
val regParam = 0.1
val initialWeightsWithIntercept = Vectors.dense(new Array[Double](numFeatures + 1))
//计算LBFGS
val (weightsWithIntercept, loss) = LBFGS.runLBFGS(
  training,
  new LogisticGradient(),
  new SquaredL2Updater(),
  numCorrections,
  convergenceTol,
  maxNumIterations,
  regParam,
  initialWeightsWithIntercept)
val model = new LogisticRegressionModel(
  Vectors.dense(weightsWithIntercept.toArray.slice(0, weightsWithIntercept.size - 1)),
  weightsWithIntercept(weightsWithIntercept.size - 1))
// Clear the default threshold.
model.clearThreshold()
// Compute raw scores on the test set.
val scoreAndLabels = test.map { point =>
  val score = model.predict(point.features)
  (score, point.label)
}
// Get evaluation metrics.
val metrics = new BinaryClassificationMetrics(scoreAndLabels)
val auROC = metrics.areaUnderROC()
loss.foreach(println)
```

### 3.2 算法实现

&emsp;&emsp;通过上文的实例，`LBFGS`通过方法`LBFGS.runLBFGS`来实现。我们来看这个入口函数。

```scala
 def runLBFGS(
      data: RDD[(Double, Vector)],
      gradient: Gradient,
      updater: Updater,
      numCorrections: Int,
      convergenceTol: Double,
      maxNumIterations: Int,
      regParam: Double,
      initialWeights: Vector): (Vector, Array[Double]) = {
    val lossHistory = mutable.ArrayBuilder.make[Double]
    val numExamples = data.count()
    //计算梯度和损失
    val costFun = new CostFun(data, gradient, updater, regParam, numExamples)
    val lbfgs = new BreezeLBFGS[BDV[Double]](maxNumIterations, numCorrections, convergenceTol)
    val states =
      lbfgs.iterations(new CachedDiffFunction(costFun), initialWeights.toBreeze.toDenseVector)
    var state = states.next()
    while (states.hasNext) {
      lossHistory += state.value
      state = states.next()
    }
    lossHistory += state.value
    val weights = Vectors.fromBreeze(state.x)
    val lossHistoryArray = lossHistory.result()
    (weights, lossHistoryArray)
  }
```

&emsp;&emsp;上文的`CostFun`类用于计算梯度和损失函数的值,我们在梯度下降算法中有介绍。`lbfgs.iterations`用于计算权重。下面分别分析这两部分。

#### 3.2.1 CostFun

```scala
private class CostFun(
    data: RDD[(Double, Vector)],
    gradient: Gradient,
    updater: Updater,
    regParam: Double,
    numExamples: Long) extends DiffFunction[BDV[Double]] {
    override def calculate(weights: BDV[Double]): (Double, BDV[Double]) = {
      // Have a local copy to avoid the serialization of CostFun object which is not serializable.
      val w = Vectors.fromBreeze(weights)
      val n = w.size
      val bcW = data.context.broadcast(w)
      val localGradient = gradient
      //通过localGradient.compute计算梯度和损失值
      val (gradientSum, lossSum) = data.treeAggregate((Vectors.zeros(n), 0.0))(
          seqOp = (c, v) => (c, v) match { case ((grad, loss), (label, features)) =>
            val l = localGradient.compute(
              features, label, bcW.value, grad)
            (grad, loss + l)
          },
          combOp = (c1, c2) => (c1, c2) match { case ((grad1, loss1), (grad2, loss2)) =>
            axpy(1.0, grad2, grad1)
            (grad1, loss1 + loss2)
          })
      //更新权重并计算正则化值
      val regVal = updater.compute(w, Vectors.zeros(n), 0, 1, regParam)._2
      val loss = lossSum / numExamples + regVal
      val gradientTotal = w.copy
      //更新权重
      axpy(-1.0, updater.compute(w, Vectors.zeros(n), 1, 1, regParam)._1, gradientTotal)
      // gradientTotal = gradientSum / numExamples + gradientTotal
      axpy(1.0 / numExamples, gradientSum, gradientTotal)
      (loss, gradientTotal.toBreeze.asInstanceOf[BDV[Double]])
    }
}
```
&emsp;&emsp;`localGradient.compute`用于计算每个样本的梯度。不同的损失函数的实现不同。查看[逻辑回归](../../分类和回归/线性模型/逻辑回归/logic-regression.md)了解`LogisticGradient`的实现。
`updater.compute`用于更新权重值并计算正则化值。最常用的正则化函数是`L2`，即下文的`SquaredL2Updater`。

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
    val thisIterStepSize = stepSize / math.sqrt(iter)
    val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
    //正则化
    brzWeights :*= (1.0 - thisIterStepSize * regParam)
    //y += x * a，即brzWeights -= gradient * thisInterStepSize
    brzAxpy(-thisIterStepSize, gradient.toBreeze, brzWeights)
    //norm
    val norm = brzNorm(brzWeights, 2.0)
    (Vectors.fromBreeze(brzWeights), 0.5 * regParam * norm * norm)
  }
}
```
&emsp;&emsp;`compute`方法实现的逻辑就是`w' = w - thisIterStepSize * (gradient + regParam * w)`。所以，根据`stepSize`不同，计算得到的值有区别。
当调用`updater.compute(w, Vectors.zeros(n), 0, 1, regParam)`，即`stepSize`为0，`iter`为1时，`regVal`即为权重平方之和`norm * norm`。否则权重需要减去`thisIterStepSize * (gradient + regParam * w)`。

#### 3.2.2 BreezeLBFGS

&emsp;&emsp;`BreezeLBFGS`使用上文分析的`CostFun`计算梯度并迭代更新权重。

```scala
val lbfgs = new BreezeLBFGS[BDV[Double]](maxNumIterations, numCorrections, convergenceTol)
val states =
      lbfgs.iterations(new CachedDiffFunction(costFun), initialWeights.toBreeze.toDenseVector)
```
&emsp;&emsp;下面重点分析`lbfgs.iterations`的实现。

```scala
def iterations(f: DF, init: T): Iterator[State] = {
    val adjustedFun = adjustFunction(f)
    infiniteIterations(f, initialState(adjustedFun, init)).takeUpToWhere(_.converged)
}
//调用infiniteIterations，其中State是一个样本类
def infiniteIterations(f: DF, state: State): Iterator[State] = {
    var failedOnce = false
    val adjustedFun = adjustFunction(f)
    //无限迭代
    Iterator.iterate(state) { state => try {
        //1 选择梯度下降方向
        val dir = chooseDescentDirection(state, adjustedFun)
        //2 计算步长
        val stepSize = determineStepSize(state, adjustedFun, dir)
        //3 更新权重
        val x = takeStep(state,dir,stepSize)
        //4 利用CostFun.calculate计算损失值和梯度
        val (value,grad) = calculateObjective(adjustedFun, x, state.history)
        val (adjValue,adjGrad) = adjust(x,grad,value)
        val oneOffImprovement = (state.adjustedValue - adjValue)/(state.adjustedValue.abs max adjValue.abs max 1E-6 * state.initialAdjVal.abs)
        //5 计算s和t
        val history = updateHistory(x,grad,value, adjustedFun, state)
        //6 只保存m个需要的s和t
        val newAverage = updateFValWindow(state, adjValue)
        failedOnce = false
        var s = State(x,value,grad,adjValue,adjGrad,state.iter + 1, state.initialAdjVal, history, newAverage, 0)
        val improvementFailure = (state.fVals.length >= minImprovementWindow && state.fVals.nonEmpty && state.fVals.last > state.fVals.head * (1-improvementTol))
        if(improvementFailure)
          s = s.copy(fVals = IndexedSeq.empty, numImprovementFailures = state.numImprovementFailures + 1)
        s
      } catch {
        case x: FirstOrderException if !failedOnce =>
          failedOnce = true
          logger.error("Failure! Resetting history: " + x)
          state.copy(history = initialHistory(adjustedFun, state.x))
        case x: FirstOrderException =>
          logger.error("Failure again! Giving up and returning. Maybe the objective is just poorly behaved?")
          state.copy(searchFailed = true)
      }
    }
  }
```
&emsp;&emsp;看上面的代码注释，它的流程可以分五步来分析。

- **1** 选择梯度下降方向

```scala
protected def chooseDescentDirection(state: State, fn: DiffFunction[T]):T = {
    state.history * state.grad
}
```
&emsp;&emsp;这里的`*`是重写的方法，它的实现如下：

```scala
def *(grad: T) = {
     val diag = if(historyLength > 0) {
       val prevStep = memStep.head
       val prevGradStep = memGradDelta.head
       val sy = prevStep dot prevGradStep
       val yy = prevGradStep dot prevGradStep
       if(sy < 0 || sy.isNaN) throw new NaNHistory
       sy/yy
     } else {
       1.0
     }
     val dir = space.copy(grad)
     val as = new Array[Double](m)
     val rho = new Array[Double](m)
     //第一次递归
     for(i <- 0 until historyLength) {
       rho(i) = (memStep(i) dot memGradDelta(i))
       as(i) = (memStep(i) dot dir)/rho(i)
       if(as(i).isNaN) {
         throw new NaNHistory
       }
       axpy(-as(i), memGradDelta(i), dir)
     }
     dir *= diag
     //第二次递归
     for(i <- (historyLength - 1) to 0 by (-1)) {
       val beta = (memGradDelta(i) dot dir)/rho(i)
       axpy(as(i) - beta, memStep(i), dir)
     }
     dir *= -1.0
     dir
    }
  }
```
&emsp;&emsp;非常明显，该方法就是实现了上文提到的`two-loop recursion`算法。

- **2** 计算步长

```scala
protected def determineStepSize(state: State, f: DiffFunction[T], dir: T) = {
    val x = state.x
    val grad = state.grad
    val ff = LineSearch.functionFromSearchDirection(f, x, dir)
    val search = new StrongWolfeLineSearch(maxZoomIter = 10, maxLineSearchIter = 10) // TODO: Need good default values here.
    val alpha = search.minimize(ff, if(state.iter == 0.0) 1.0/norm(dir) else 1.0)
    if(alpha * norm(grad) < 1E-10)
      throw new StepSizeUnderflow
    alpha
  }
```
&emsp;&emsp;这一步对应`L-BFGS`的步骤的`Step 5`，通过一维搜索计算步长。

- **3** 更新权重

```scala
protected def takeStep(state: State, dir: T, stepSize: Double) = state.x + dir * stepSize
```
&emsp;&emsp;这一步对应`L-BFGS`的步骤的`Step 5`，更新权重。

- **4** 计算损失值和梯度

```scala
 protected def calculateObjective(f: DF, x: T, history: History): (Double, T) = {
     f.calculate(x)
  }
```
&emsp;&emsp;这一步对应`L-BFGS`的步骤的`Step 7`，利用上文介绍的`CostFun.calculate`计算梯度和损失值。并计算出`s`和`t`。

- **5** 计算s和t，并更新history

```scala
//计算s和t
protected def updateHistory(newX: T, newGrad: T, newVal: Double,  f: DiffFunction[T], oldState: State): History = {
    oldState.history.updated(newX - oldState.x, newGrad :- oldState.grad)
}
//添加新的s和t，并删除过期的s和t
protected def updateFValWindow(oldState: State, newAdjVal: Double):IndexedSeq[Double] = {
    val interm = oldState.fVals :+ newAdjVal
    if(interm.length > minImprovementWindow) interm.drop(1)
    else interm
  }
```

# 参考文献

【1】陈宝林，最优化理论和算法

【2】[Updating Quasi-Newton Matrices with Limited Storage](docs/Updating  Quasi-Newton  Matrices  with  Limited  Storage.pdf)

【3】[On the Limited Memory BFGS Method for Large Scale Optimization](docs/On the Limited Memory BFGS Method for Large Scale Optimization.pdf)

【4】[L-BFGS算法](http://blog.csdn.net/acdreamers/article/details/44728041)

【5】[BFGS算法](http://wenku.baidu.com/link?url=xyN5e-LMR2Ztq90-J95oKHUFBLP8gkLzlbFI6ptbgXMWYt5xTZHgXexWcbjQUmGahQpr39AIc0AomDeFqyY7mn7VqLoQj6gcDHDOccJGln3)

【6】[逻辑回归模型及LBFGS的Sherman Morrison(SM) 公式推导](http://blog.csdn.net/zhirom/article/details/38332111)


















