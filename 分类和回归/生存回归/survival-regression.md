# 生存回归

## 1 基本概念

### 1.1 生存数据

&emsp;&emsp;生存数据就是关于某个体生存时间的数据。生存时间就是死亡时间减去出生时间。例如，以一个自然人的出生为“出生”，死亡为“死亡”。
那么，死亡时间减去出生时间，就是一个人的寿命，这是一个典型的生存数据。类似的例子，还可以举出很多。所有这些数据都有一个共同的特点，
就是需要清晰定义的：出生和死亡 。如果用死亡时间减去出生时间，就产生了一个生存数据。因为死亡一定发生在出生的后面，因此，生存数据一定是正数。
因为，从理论上讲，出生死亡时间都可能取任意数值，因此 生存数据一定是连续的正数。

&emsp;&emsp;生存期不同于一般指标，他有二个特点：

- 1 有截尾数据（`censored data`)

&emsp;&emsp;例如我们在疾病预测的实验中，随访未能知道病人的确切生存时间，只知道病人的生存时间大于某时间。

（1）病人失访或因其他原因而死亡---失访
（2）到了研究的终止期病人尚未死亡---终访

&emsp;&emsp;例如，一个人的寿命。假设我关心`1949`年出生的人群的平均寿命。这群人可以被分成两部分。一部分是已经离世了，所以他们的死亡时间是准确知道的。因此，他们的寿命是非常清晰的。
另一部分，是所有健在的人群，他们从`1949`年出生到现在，已经走过了将近70个春秋岁月，但是他们还活着！到`2017`年为止，他们已经生存了`68`年，但是他们最终的寿命是多少？我们是不知道的。
我们知道他们的寿命一定会比`68`大，数学上可以被记作`68+`。但是，到底“+”多少，不清楚。

&emsp;&emsp;虽然截尾数据提供的信息是不完全的，但不能删去，因为这不仅损失了资料，而且会造成偏性。

- 2  生存时间的特征一般不服从正态分布

&emsp;&emsp;跟所有的数据分析一样，要分析生存数据，首要问题是做描述性分析。如果生存数据没有被截断，那么所有常规的描述统计量，估计量都适用。例如：样本均值，样本方差等。
但是，如果生存数据存在大量的截断数据，那么任何同均值相关的统计量就都没法计算了。例如：样本均值无法算，样本方差涉及到因变量的平方的均值，因此它也没法计算。

&emsp;&emsp;真实的数据常常非常复杂，每个样本的出生日期不同，死亡日期不同，截断时间点不同。但是，不管这个数据如何复杂，其背后的基本原理是一样的。
那就是：虽然样本均值没法估计，样本方差没法估计。但是，各种分位数却在一个很大的范围内可以被估计。如果这个范围大到可以覆盖中位数，那么从某种意义上讲，我们也就把握了生存的平均状况了。

&emsp;&emsp;总结一下就是：对生存数据最基本的描述分析方法，不是过去常见的样本均值，样本方差等等，而是各种分位数。这些分位数也就构成了所谓的生存函数。生存函数就变成了对生存数据最基本的描述统计。

### 1.2 描述生存时间分布规律的函数

- 1 生存率(`Survival Rate`)

&emsp;&emsp;又称为生存概率或生存函数，它表示生存时间长于时间`t`的概率，用`S(t)` 表示：`s(t)=P（T≥t）`。以时间`t`为横坐标，`S(t)`为纵坐标所作的曲线称为生存率曲线，它是一条下降的曲线，下降的坡度越陡，
表示生存率越低或生存时间越短，其斜率表示死亡速率。

- 2 概率密度函数(`Probability Density Function`)

&emsp;&emsp;其定义为:`f(t)=lim (一个病人在区间(t,t+△t)内死亡概率/△t)`，它表示死亡速率的大小。如以`t`为横坐，`f(t)`为纵坐标作出的曲线称为密度曲线，由曲线上可看出不同时间的死亡速率及死亡高峰时间。
纵坐标越大，其死亡速率越高，如曲线呈现单调下降，则死亡速率越来越小，如呈现峰值，则为死亡高峰。

- 3 风险函数(`Hazard Function`)

&emsp;&emsp;其定义为:`h(t)=lim(在时间t生存的病人死于区间(t,△t)的概率/△t)`，由于计算`h(t)`时，用到了生存到时间`t`这一条件，故上式极限式中分子部分是一个条件概率。
可将`h(t)`称为生存到时间`t`的病人在时间`t`的瞬时死亡率或条件死亡速率或年龄别死亡速率。当用`t`作横坐标，`h(t)`为纵坐标所绘的曲线，如递增，则表示条件死亡速率随时间而增加，如平行于横轴，
则表示没有随时间而加速(或减少)死亡的情况。

## 2 加速失效时间模型(AFT)

&emsp;&emsp;在生存分析领域，加速失效时间模型(`accelerated failure time model`,`AFT` 模型)可以作为比例风险模型的替代模型。`AFT`模型将线性回归模型的建模方法引人到生存分析的领域，
将生存时间的对数作为反应变量，研究多协变量与对数生存时间之间的回归关系，在形式上，模型与一般的线性回归模型相似。对回归系数的解释也与一般的线性回归模型相似，较之`Cox`模型，
`AFT`模型对分析结果的解释更加简单、直观且易于理解，并且可以预测个体的生存时间。

&emsp;&emsp;在`spark ml`中，实现了`AFT` 模型，这是一个用于检查数据的参数生存回归模型。它描述了生存时间对数的模型，因此它通常被称为生存分析的对数线性模型。不同于为相同目的设计的比例风险模型(`Proportional hazards model`)，
`AFT`模型更容易并行化，因为每个实例独立地贡献于目标函数。

&emsp;&emsp;给定给定协变量的值$x^{'}$，对于`i = 1, …, n`可能的右截尾的随机生存时间$t_{i}$，`AFT`模型的似然函数如下：

$$L(\beta,\sigma)=\prod_{i=1}^n[\frac{1}{\sigma}f_{0}(\frac{\log{t_{i}}-x^{'}\beta}{\sigma})]^{\delta_{i}}S_{0}(\frac{\log{t_{i}}-x^{'}\beta}{\sigma})^{1-\delta_{i}}$$

&emsp;&emsp;其中，$\delta_{i}$是指示器，它表示事件`i`是否发生了，即有无截尾。使$\epsilon_{i}=\frac{\log{t_{i}}-x^{‘}\beta}{\sigma}$，则对数似然函数为以下形式：

$$\iota(\beta,\sigma)=\sum_{i=1}^{n}[-\delta_{i}\log\sigma+\delta_{i}\log{f_{0}}(\epsilon_{i})+(1-\delta_{i})\log{S_{0}(\epsilon_{i})}]$$

&emsp;&emsp;其中$S_{0}(\epsilon_{i})$是基准生存函数，$f_{0}(\epsilon_{i})$是对应的概率密度函数。

&emsp;&emsp;最常用的`AFT`模型基于服从韦伯分布的生存时间，生存时间的韦伯分布对应于生存时间对数的极值分布，所以$S_{0}(\epsilon)$函数为：

$$S_{0}(\epsilon_{i})=\exp(-e^{\epsilon_{i}})$$

&emsp;&emsp;$f_{0}(\epsilon_{i})$函数为：

$$f_{0}(\epsilon_{i})=e^{\epsilon_{i}}\exp(-e^{\epsilon_{i}})$$

&emsp;&emsp;生存时间服从韦伯分布的`AFT`模型的对数似然函数如下：

$$\iota(\beta,\sigma)= -\sum_{i=1}^n[\delta_{i}\log\sigma-\delta_{i}\epsilon_{i}+e^{\epsilon_{i}}]$$

&emsp;&emsp;由于最小化对数似然函数的负数等于最大化后验概率，所以我们要优化的损失函数为$-\iota(\beta,\sigma)$。分别对$\beta$和$\log\sigma$求导，得到：

$$\frac{\partial (-\iota)}{\partial \beta}=\sum_{1=1}^{n}[\delta_{i}-e^{\epsilon_{i}}]\frac{x_{i}}{\sigma}$$

$$\frac{\partial (-\iota)}{\partial (\log\sigma)}=\sum_{i=1}^{n}[\delta_{i}+(\delta_{i}-e^{\epsilon_{i}})\epsilon_{i}]$$

&emsp;&emsp;可以证明`AFT`模型是一个凸优化问题，即是说找到凸函数$-\iota(\beta,\sigma)$的最小值取决于系数向量$\beta$以及尺度参数的对数$\log\sigma$。
`spark ml`中使用`L-BFGS`作为优化算法。

>>> 注意：当使用无拦截(`intercept`)的连续非零列训练`AFTSurvivalRegressionModel`时，`Spark MLlib`为连续非零列输出零系数。这种处理与R中的生存函数`survreg`不同。

## 3 例子

```scala
 val dataList: List[(Double, Double, Double, Double)] = List(
      (2, 51, 1, 1),
      (2, 58, 1, 1),
      (2, 55, 2, 1),
      (2, 28, 22, 1),
      (1, 21, 30, 0),
      (1, 19, 28, 1),
      (2, 25, 32, 1),
      (2, 48, 11, 1),
      (2, 47, 14, 1),
      (2, 25, 36, 0),
      (2, 31, 31, 0),
      (1, 24, 33, 0),
      (1, 25, 33, 0),
      (2, 30, 37, 0),
      (2, 33, 35, 0),
      (1, 36, 25, 1),
      (1, 30, 31, 0),
      (1, 41, 22, 1),
      (2, 43, 26, 1),
      (2, 45, 24, 1),
      (2, 35, 35, 0),
      (1, 29, 34, 0),
      (1, 35, 30, 0),
      (1, 32, 35, 1),
      (2, 36, 40, 1),
      (1, 32, 39, 0))
    val data = dataList.toDF("sex", "age", "label", "censor").orderBy("label")
    val colArray = Array("sex", "age")
    val assembler = new VectorAssembler().setInputCols(colArray).setOutputCol("features")
    val vecDF: DataFrame = assembler.transform(data)
    val aft = new AFTSurvivalRegression()
    val model = aft.fit(vecDF)
    // Print the coefficients, intercept and scale parameter for AFT survival regression
    println(s"Coefficients: ${model.coefficients} Intercept: " +
      s"${model.intercept} Scale: ${model.scale}")
    val Array(coeff1, coeff2) = model.coefficients.toArray
    val intercept: Double = model.intercept
    val scale: Double = model.scale
    val aftDF = model.transform(vecDF)
    // 风险率h(t)
    aftDF.selectExpr("sex", "age", "label", "censor",
      "features", "round(prediction,2) as prediction",
      s"round( exp( sex*$coeff1+age*$coeff2+$intercept ), 2) as h(t)").orderBy("label").show(100, false)
```

## 4 参考文献

【1】[Spark Doc](https://spark.apache.org/docs/latest/ml-classification-regression.html#survival-regression)

【2】[回归XY | 数据江湖：回归五式之第五式（生存回归）](https://www.wxzhi.com/archives/871/pj2zikqb49cof749/)
