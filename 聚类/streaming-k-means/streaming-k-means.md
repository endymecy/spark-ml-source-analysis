# 流式`k-means`算法

&emsp;&emsp;当数据是以流的方式到达的时候，我们可能想动态的估计（`estimate `）聚类的簇，通过新的到达的数据来更新聚类。`spark.mllib`支持流式`k-means`聚类，并且可以通过参数控制估计衰减（`decay`）(或“健忘”(`forgetfulness`))。
这个算法使用一般地小批量更新规则来更新簇。

## 流式`k-means`算法原理

&emsp;&emsp;对每批新到的数据，我们首先将所有的点分配给距离它们最近的簇，然后计算新的数据中心，最后更新每一个簇。使用的公式如下所示：

<div  align="center"><img src="imgs/streaming-k-means.1.1.png" width = "400" height = "50" alt="1.1" align="center" /></div><br />

<div  align="center"><img src="imgs/streaming-k-means.1.2.png" width = "380" height = "25" alt="1.2" align="center" /></div><br />


