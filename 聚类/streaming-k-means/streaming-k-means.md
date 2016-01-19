# 流式`k-means`算法

&emsp;&emsp;当数据是以流的方式到达的时候，我们可能想动态的估计（`estimate `）聚类的簇，通过新的到达的数据来更新聚类。`spark.mllib`支持流式`k-means`聚类，并且可以通过参数控制估计衰减（`decay`）(或“健忘”(`forgetfulness`))。
这个算法使用一般地小批量更新规则来更新簇。

## 1 流式`k-means`算法原理

&emsp;&emsp;对每批新到的数据，我们首先将点分配给距离它们最近的簇，然后计算新的数据中心，最后更新每一个簇。使用的公式如下所示：

<div  align="center"><img src="imgs/streaming-k-means.1.1.png" width = "400" height = "50" alt="1.1" align="center" /></div><br />

<div  align="center"><img src="imgs/streaming-k-means.1.2.png" width = "380" height = "25" alt="1.2" align="center" /></div><br />

&emsp;&emsp;在上面的公式中，<img src="http://www.forkosh.com/mathtex.cgi?{c}_{t}">表示前一个簇中心，<img src="http://www.forkosh.com/mathtex.cgi?{n}_{t}">表示分配给这个簇的点的数量，
<img src="http://www.forkosh.com/mathtex.cgi?{x}_{t}">表示从当前批数据的簇中心，<img src="http://www.forkosh.com/mathtex.cgi?{m}_{t}">表示当前批数据的点数量。
当评价新的数据时，把衰减因子`alpha`当做折扣加权应用到当前的点上，用以衡量当前预测的簇的贡献度量。当`alpha`等于1时，所有的批数据赋予相同的权重，当`alpha`等于0时，数据中心点完全通过当前数据确定。

&emsp;&emsp;衰减因子`alpha`也可以通过`halfLife`参数联合时间单元（`time unit`）来确定，时间单元可以是一批数据也可以是一个数据点。假如数据从`t`时刻到来并定义了`halfLife`为`h`，
在`t+h`时刻，应用到`t`时刻的数据的折扣（`discount`）为0.5。

&emsp;&emsp;流式`k-means`算法的步骤如下所示：

- （1）分配新的数据点到离其最近的簇；

- （2）根据时间单元（`time unit`）计算折扣（`discount`）值，并更新簇权重；

- （3）应用更新规则；

- （4）应用更新规则后，有些簇可能消失了，那么切分最大的簇为两个簇。

## 2 流式`k-means`算法源码分析

&emsp;&emsp;在分步骤分析源码之前，我们先了解一下`StreamingKMeans`参数表达的含义。

```scala
class StreamingKMeans(
    var k: Int, //簇个数
    var decayFactor: Double,//衰减因子
    var timeUnit: String //时间单元
)
```
&emsp;&emsp;在上述定义中，`k`表示我们要聚类的个数，`decayFactor`表示衰减因子，用于计算折扣，`timeUnit`表示时间单元，时间单元既可以是一批数据（`StreamingKMeans.BATCHES`）也可以是单条数据（`StreamingKMeans.POINTS`）。

&emsp;&emsp;由于我们处理的是流式数据，所以我们在流式数据来之前要先初始化模型。有两种初始化模型的方法，一种是直接指定初始化中心点及簇权重，一种是随机初始化中心点以及簇权重。

```scala
 //直接初始化中心点及簇权重
 def setInitialCenters(centers: Array[Vector], weights: Array[Double]): this.type = {
    model = new StreamingKMeansModel(centers, weights)
    this
 }
 //随机初始化中心点以及簇权重
 def setRandomCenters(dim: Int, weight: Double, seed: Long = Utils.random.nextLong): this.type = {
     val random = new XORShiftRandom(seed)
     val centers = Array.fill(k)(Vectors.dense(Array.fill(dim)(random.nextGaussian())))
     val weights = Array.fill(k)(weight)
     model = new StreamingKMeansModel(centers, weights)
     this
 } 
```
&emsp;&emsp;初始化中心点以及簇权重之后，对于新到的流数据，我们使用更新规则修改中心点和权重，调整聚类情况。更新过程在`update`方法中实现，下面我们分步骤分析该方法。

- 