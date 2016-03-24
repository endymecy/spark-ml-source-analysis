# 假设检测

&emsp;&emsp;假设检测是统计中有力的工具，它用于判断一个结果是否在统计上是显著的、这个结果是否有机会发生。`spark.mllib`目前支持皮尔森卡方检测。输入属性的类型决定是作拟合优度(`goodness of fit`)检测还是作独立性检测。
拟合优度检测需要输入数据的类型是`vector`，独立性检测需要输入数据的类型是`Matrix`。

&emsp;&emsp;`spark.mllib`也支持输入数据类型为`RDD[LabeledPoint]`，它用来通过卡方独立性检测作特征选择。`Statistics`提供方法用来作皮尔森卡方检测。下面是一个例子：

```scala
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.Statistics._
val sc: SparkContext = ...
val vec: Vector = ... // a vector composed of the frequencies of events
// 作皮尔森拟合优度检测 
val goodnessOfFitTestResult = Statistics.chiSqTest(vec)
println(goodnessOfFitTestResult) 
val mat: Matrix = ... // a contingency matrix
// 作皮尔森独立性检测
val independenceTestResult = Statistics.chiSqTest(mat) 
println(independenceTestResult) // summary of the test including the p-value, degrees of freedom...
val obs: RDD[LabeledPoint] = ... // (feature, label) pairs.
// 独立性检测用于特征选择
val featureTestResults: Array[ChiSqTestResult] = Statistics.chiSqTest(obs)
var i = 1
featureTestResults.foreach { result =>
    println(s"Column $i:\n$result")
    i += 1
}
```
&emsp;&emsp;另外，`spark.mllib`提供了一个`Kolmogorov-Smirnov (KS)`检测的`1-sample, 2-sided`实现，用来检测概率分布的相等性。通过提供理论分布（现在仅仅支持正太分布）的名字以及它相应的参数，
或者提供一个计算累积分布(`cumulative distribution`)的函数，用户可以检测原假设或零假设(`null hypothesis`)：即样本是否来自于这个分布。用户检测正太分布，但是不提供分布参数，检测会默认该分布为标准正太分布。

&emsp;&emsp;`Statistics`提供了一个运行`1-sample, 2-sided KS`检测的方法，下面就是一个应用的例子。

```scala
import org.apache.spark.mllib.stat.Statistics
val data: RDD[Double] = ... // an RDD of sample data
// run a KS test for the sample versus a standard normal distribution
val testResult = Statistics.kolmogorovSmirnovTest(data, "norm", 0, 1)
println(testResult) 
// perform a KS test using a cumulative distribution function of our making
val myCDF: Double => Double = ...
val testResult2 = Statistics.kolmogorovSmirnovTest(data, myCDF)
```

## 流式显著性检测

&emsp;&emsp;显著性检验即用于实验处理组与对照组或两种不同处理的效应之间是否有差异，以及这种差异是否显著的方法。

&emsp;&emsp;常把一个要检验的假设记作`H0`,称为原假设（或零假设） (`null hypothesis`) ，与`H0`对立的假设记作`H1`，称为备择假设(`alternative hypothesis`) 。

- 在原假设为真时，决定放弃原假设，称为第一类错误，其出现的概率通常记作`alpha`

- 在原假设不真时，决定接受原假设，称为第二类错误，其出现的概率通常记作`beta`

&emsp;&emsp;通常只限定犯第一类错误的最大概率`alpha`， 不考虑犯第二类错误的概率`beta`。这样的假设检验又称为显著性检验，概率`alpha`称为显著性水平。

&emsp;&emsp;`MLlib`提供一些检测的在线实现，用于支持诸如`A/B`测试的场景。这些检测可能执行在`Spark Streaming`的`DStream[(Boolean,Double)]`上，元组的第一个元素表示控制组(`control group (false)`)或者处理组(` treatment group (true)`),
第二个元素表示观察者的值。

&emsp;&emsp;流式显著性检测支持下面的参数：

- `peacePeriod`：来自流中忽略的初始数据点的数量，用于减少`novelty effects`；

- `windowSize`：执行假设检测的以往批次的数量。如果设置为0，将对之前所有的批次数据作累积处理。

&emsp;&emsp;`StreamingTest`支持流式假设检测。下面是一个应用的例子。

```scala
val data = ssc.textFileStream(dataDir).map(line => line.split(",") match {
  case Array(label, value) => BinarySample(label.toBoolean, value.toDouble)
})
val streamingTest = new StreamingTest()
  .setPeacePeriod(0)
  .setWindowSize(0)
  .setTestMethod("welch")
val out = streamingTest.registerStream(data)
out.print()
```

# 参考文献

【1】[显著性检验](http://wiki.mbalib.com/wiki/Significance_Testing)