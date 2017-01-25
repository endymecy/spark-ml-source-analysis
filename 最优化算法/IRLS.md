# 迭代再加权最小二乘

## 1 原理

&emsp;&emsp;迭代再加权最小二乘(`IRLS`)用于解决特定的最优化问题，这个最优化问题的目标函数如下所示：

$$arg min_{\beta} \sum_{i=1}^{n}|y_{i} - f_{i}(\beta)|^{p}$$

&emsp;&emsp;这个目标函数可以通过迭代的方法求解。在每次迭代中，解决一个带权最小二乘问题，形式如下：

$$\beta ^{t+1} = argmin_{\beta} \sum_{i=1}^{n} w_{i}(\beta^{(t)}))|y_{i} - f_{i}(\beta)|^{2} = (X^{T}W^{(t)}X)^{-1}X^{T}W^{(t)}y$$

&emsp;&emsp;在这个公式中，$W^{(t)}$是权重对角矩阵，它的所有元素都初始化为1。每次迭代中，通过下面的公式更新。

$$W_{i}^{(t)} = |y_{i} - X_{i}\beta^{(t)}|^{p-2}$$

## 2 源码分析

&emsp;&emsp;在`spark ml`中，迭代再加权最小二乘主要解决广义线性回归问题。下面看看实现代码。

### 2.1 更新权重

```scala
 // Update offsets and weights using reweightFunc
 val newInstances = instances.map { instance =>
    val (newOffset, newWeight) = reweightFunc(instance, oldModel)
    Instance(newOffset, newWeight, instance.features)
 }
```
&emsp;&emsp;这里使用`reweightFunc`方法更新权重。具体的实现在广义线性回归的实现中。

```scala
    /**
     * The reweight function used to update offsets and weights
     * at each iteration of [[IterativelyReweightedLeastSquares]].
     */
    val reweightFunc: (Instance, WeightedLeastSquaresModel) => (Double, Double) = {
      (instance: Instance, model: WeightedLeastSquaresModel) => {
        val eta = model.predict(instance.features)
        val mu = fitted(eta)
        val offset = eta + (instance.label - mu) * link.deriv(mu)
        val weight = instance.weight / (math.pow(this.link.deriv(mu), 2.0) * family.variance(mu))
        (offset, weight)
      }
    }
    
    def fitted(eta: Double): Double = family.project(link.unlink(eta))
```
&emsp;&emsp;这里的`model.predict`利用带权最小二乘模型预测样本的取值，然后调用`fitted`方法计算均值函数$\mu$。`offset`表示
更新后的标签值，`weight`表示更新后的权重。关于链接函数的相关计算可以参考[广义线性回归](../分类和回归/线性模型/广义线性回归/glr.md)的分析。

&emsp;&emsp;有一点需要说明的是，这段代码中标签和权重的更新并没有参照上面的原理或者说我理解有误。

### 2.2 训练新的模型

```scala
  // 使用更新过的样本训练新的模型 
  model = new WeightedLeastSquares(fitIntercept, regParam, elasticNetParam = 0.0,
        standardizeFeatures = false, standardizeLabel = false).fit(newInstances)

  // 检查是否收敛
  val oldCoefficients = oldModel.coefficients
  val coefficients = model.coefficients
  BLAS.axpy(-1.0, coefficients, oldCoefficients)
  val maxTolOfCoefficients = oldCoefficients.toArray.reduce { (x, y) =>
        math.max(math.abs(x), math.abs(y))
  }
  val maxTol = math.max(maxTolOfCoefficients, math.abs(oldModel.intercept - model.intercept))
  if (maxTol < tol) {
    converged = true
  }
```
&emsp;&emsp;训练完新的模型后，重复2.1步，直到参数收敛或者到达迭代的最大次数。

## 3 参考文献

【1】[Iteratively reweighted least squares](https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares)
