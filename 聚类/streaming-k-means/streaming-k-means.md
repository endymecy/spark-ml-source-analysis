# 流式`k-means`算法

&emsp;&emsp;当数据是以流的方式到达的时候，我们可能想动态的估计（`estimate `）聚类的簇，通过新的到达的数据来更新聚类。`spark.mllib`支持流式`k-means`聚类，并且可以通过参数控制估计衰减（`decay`）(或“健忘”(`forgetfulness`))。
这个算法使用一般地小批量更新规则来更新簇。

## 1 流式`k-means`算法原理

&emsp;&emsp;对每批新到的数据，我们首先将所有的点分配给距离它们最近的簇，然后计算新的数据中心，最后更新每一个簇。使用的公式如下所示：

<div  align="center"><img src="imgs/streaming-k-means.1.1.png" width = "400" height = "50" alt="1.1" align="center" /></div><br />

<div  align="center"><img src="imgs/streaming-k-means.1.2.png" width = "380" height = "25" alt="1.2" align="center" /></div><br />

&emsp;&emsp;在上面的公式中，<img src="http://www.forkosh.com/mathtex.cgi?{c}_{t}">表示前一个簇中心，<img src="http://www.forkosh.com/mathtex.cgi?{n}_{t}">表示分配给这个簇的点的数量，
<img src="http://www.forkosh.com/mathtex.cgi?{x}_{t}">表示从当前批数据获得的簇中心，<img src="http://www.forkosh.com/mathtex.cgi?{m}_{t}">表示当心批数据的点数量。衰减因子`alpha`用来控制对过去数据的使用：
当`alpha`等于1时，我们将使用从开始到现在所有的数据，当`alpha`等于0时，我们仅仅使用当前的数据。