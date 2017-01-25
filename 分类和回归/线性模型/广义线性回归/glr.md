# 广义线性回归

## 1 普通线性模型

&emsp;&emsp;普通线性模型(`ordinary linear model`)可以用下式表示：

$$Y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + … + \beta_{p-1} x_{p-1} + \epsilon$$

&emsp;&emsp;这里$\beta$是未知参数，$\epsilon$是误差项。普通线性模型主要有以下几点假设：

- 响应变量$Y$和误差项$\epsilon$均服从正太分布。其中$\epsilon \sim N(0,{{\sigma }^{2}})$，$Y\sim N({{\beta }^{T}}x,{{\sigma }^{2}})$。
- 预测量$x_i$和未知参数$\beta_i$均具有非随机性。预测量$x_i$具有非随机性、可测且不存在测量误差；未知参数$\beta_i$被认为是未知但不具随机性的常数。
- 普通线性模型的输出项是随机变量$Y$。普通线性模型主要研究响应变量的期望$E[Y]$。
- 连接方式：在上面三点假设下，对上式两边取数学期望，可得

$$E[Y] = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + … + \beta_{p-1} x_{p-1}$$

&emsp;&emsp;在普通线性模型里，响应变量的均值$E[Y]$与预测量的线性组合$\beta_0 + \beta_1 x_1 + \beta_2 x_2 + … + \beta_{p-1} x_{p-1}$通过恒等式(`identity`)连接。
也可以说是通过$f(x)=x$这个链接函数(`link function`)连接。

## 2 广义线性模型

&emsp;&emsp;广义线性模型(`generalized linear model`)是在普通线性模型的基础上，对上述四点假设进行推广而得出的应用范围更广，更具实用性的回归模型。
主要有两点不同，这两点分别是：

- 响应变量$Y$和误差项$\epsilon$的分布推广至指数分散族(`exponential dispersion family`)。在`spark ml`中，广义线性回归支持的指数分布分别是正态分布、泊松分布、二项分布以及伽玛分布。
- 连接方式：广义线性模型里采用的链接函数(`link function`)理论上可以是任意的，而不再局限于$f(x)=x$。

&emsp;&emsp;这里需要重点说明一下链接函数。链接函数描述了线性预测$X\beta$与分布期望值$E[Y]$的关系：$E[Y] = \mu = g^{-1}(X\beta)$，其中$g$表示链接函数，$\mu$表示均值函数。
一般情况下，高斯分布对应于恒等式，泊松分布对应于自然对数函数等。下面列出了`spark ml`中提供的链接函数以及该链接函数使用的指数分布。

| 连接函数名称 | 链接函数 | 均值函数 | 对应的指数分布 | 
|------------|-------|-----------|-------------|
| identity（恒等）| $\mu = X\beta$ | $\mu = X\beta$| 高斯分布，泊松分布，伽马分布 |
| inverse（倒数）| $\mu^{-1} = X\beta$ | $\mu = (X\beta)^{-1}$ | 高斯分布，伽马分布 |
| sqrt(均分) | $\mu^{1/2} = X\beta$ | $\mu = (X\beta)^{2}$ | 泊松分布 |
| log（对数）| $ln(\mu) = X\beta$ | $\mu = exp(X\beta)$ | 高斯分布，泊松分布，伽马分布 |
| logit | $ln(\frac{\mu }{1-\mu }) = X\beta$ | $\mu = \frac{exp(X\beta)}{1 + exp(1 + X\beta)}$ | 高斯分布，泊松分布，伽马分布 |
| cloglog | $ln(- ln(1-\mu)) = X\beta$ | $\mu = 1 - exp(- exp(X\beta))$ | 二次分布 |
| probit | 标准高斯分布的inverse cdf，其中p值为$\mu$ | 标准高斯分布的cdf | 二次分布 |

## 3 源码分析

### 3.1 使用实例

```scala
import org.apache.spark.ml.regression.GeneralizedLinearRegression

// Load training data
val dataset = spark.read.format("libsvm")
  .load("data/mllib/sample_linear_regression_data.txt")

val glr = new GeneralizedLinearRegression()
  .setFamily("gaussian")
  .setLink("identity")
  .setMaxIter(10)
  .setRegParam(0.3)

// Fit the model
val model = glr.fit(dataset)

// Print the coefficients and intercept for generalized linear regression model
println(s"Coefficients: ${model.coefficients}")
println(s"Intercept: ${model.intercept}")
```

### 3.2 训练模型

&emsp;&emsp;广义线性回归的训练比较简单。当指数分布是高斯分布，同时链接函数是恒等(`identity`)时，此时的情况就是普通的线性回归。可以利用带权最小二乘求解。

```scala
 val model = if (familyObj == Gaussian && linkObj == Identity) {
      val optimizer = new WeightedLeastSquares($(fitIntercept), $(regParam), elasticNetParam = 0.0,
        standardizeFeatures = true, standardizeLabel = true)
      val wlsModel = optimizer.fit(instances)
      val model = copyValues(
        new GeneralizedLinearRegressionModel(uid, wlsModel.coefficients, wlsModel.intercept)
          .setParent(this))
      val trainingSummary = new GeneralizedLinearRegressionTrainingSummary(dataset, model,
        wlsModel.diagInvAtWA.toArray, 1, getSolver)
      model.setSummary(Some(trainingSummary))
 }
```
&emsp;&emsp;如果是其它的情况，使用迭代再加权最小二乘(`Iteratively reweighted least squares(IRLS)`)求解。

```scala
// Fit Generalized Linear Model by iteratively reweighted least squares (IRLS).
   val initialModel = familyAndLink.initialize(instances, $(fitIntercept), $(regParam))
   val optimizer = new IterativelyReweightedLeastSquares(initialModel,
        familyAndLink.reweightFunc, $(fitIntercept), $(regParam), $(maxIter), $(tol))
   val irlsModel = optimizer.fit(instances)
   val model = copyValues(
     new GeneralizedLinearRegressionModel(uid, irlsModel.coefficients, irlsModel.intercept)
          .setParent(this))
   val trainingSummary = new GeneralizedLinearRegressionTrainingSummary(dataset, model,
        irlsModel.diagInvAtWA.toArray, irlsModel.numIterations, getSolver)
   model.setSummary(Some(trainingSummary))
```
&emsp;&emsp;迭代再加权最小二乘的分析见最优化章节：[迭代再加权最小二乘](../../../最优化算法/IRLS.md)。

### 3.3 链接函数

&emsp;&emsp;根据第二章中表格描述的链接函数和均值函数，我们可以很容易实现链接函数。链接函数和均值函数的值可以用于对样本进行更新，
更新相应的标签值和权重值。

- Identity

```scala
private[regression] object Identity extends Link("identity") {
    override def link(mu: Double): Double = mu  // 链接函数
    override def deriv(mu: Double): Double = 1.0  // 链接函数求导数
    override def unlink(eta: Double): Double = eta  // 均值函数
  }
```
- Logit

```scala
private[regression] object Logit extends Link("logit") {
    override def link(mu: Double): Double = math.log(mu / (1.0 - mu)) // 链接函数
    override def deriv(mu: Double): Double = 1.0 / (mu * (1.0 - mu)) // 链接函数导数
    override def unlink(eta: Double): Double = 1.0 / (1.0 + math.exp(-1.0 * eta)) // 均值函数
  }
```

- Log

```scala
  private[regression] object Log extends Link("log") {
    override def link(mu: Double): Double = math.log(mu) // 链接函数
    override def deriv(mu: Double): Double = 1.0 / mu // 链接函数导数
    override def unlink(eta: Double): Double = math.exp(eta) // 均值函数
  }
```

- Inverse

```scala
  private[regression] object Inverse extends Link("inverse") {
    override def link(mu: Double): Double = 1.0 / mu // 链接函数
    override def deriv(mu: Double): Double = -1.0 * math.pow(mu, -2.0) // 链接函数导数
    override def unlink(eta: Double): Double = 1.0 / eta // 均值函数
  }
```

- Probit

```scala
  private[regression] object Probit extends Link("probit") {
    override def link(mu: Double): Double = dist.Gaussian(0.0, 1.0).icdf(mu) // 链接函数
    override def deriv(mu: Double): Double = {
      1.0 / dist.Gaussian(0.0, 1.0).pdf(dist.Gaussian(0.0, 1.0).icdf(mu)) // 链接函数导数
    }
    override def unlink(eta: Double): Double = dist.Gaussian(0.0, 1.0).cdf(eta) // 均值函数
  }
```
- CLogLog

```scala
  private[regression] object CLogLog extends Link("cloglog") {
    override def link(mu: Double): Double = math.log(-1.0 * math.log(1 - mu)) // 链接函数
    override def deriv(mu: Double): Double = 1.0 / ((mu - 1.0) * math.log(1.0 - mu)) // 链接函数导数
    override def unlink(eta: Double): Double = 1.0 - math.exp(-1.0 * math.exp(eta)) // 均值函数
  }
```
- Sqrt

```scala
  private[regression] object Sqrt extends Link("sqrt") {
    override def link(mu: Double): Double = math.sqrt(mu) // 链接函数
    override def deriv(mu: Double): Double = 1.0 / (2.0 * math.sqrt(mu)) // 链接函数导数
    override def unlink(eta: Double): Double = eta * eta // 均值函数
  }
```

## 参考文献

【1】[从线性模型到广义线性模型](http://cos.name/2011/01/how-does-glm-generalize-lm-assumption/)

【2】[广义线性模型-维基百科](https://zh.wikipedia.org/wiki/%E5%BB%A3%E7%BE%A9%E7%B7%9A%E6%80%A7%E6%A8%A1%E5%9E%8B)