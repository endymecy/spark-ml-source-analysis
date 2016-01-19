# 快速迭代聚类

## 1 谱聚类算法的原理

&emsp;&emsp;在分析快速迭代聚类之前，我们先来了解一下谱聚类算法。谱聚类算法是建立在谱图理论的基础上的算法，与传统的聚类算法相比，它能在任意形状的样本空间上聚类且能够收敛到全局最优解。
谱聚类算法的主要思想是将聚类问题转换为无向图的划分问题。

- 首先，数据点被看做一个图的顶点`v`，两数据的相似度看做图的边，边的集合由<img src="http://www.forkosh.com/mathtex.cgi?E={A}_{ij}">表示，由此构造样本数据集的相似度矩阵`A`，并求出拉普拉斯矩阵`L`。
- 其次，根据划分准则使子图内部相似度尽量大，子图之间的相似度尽量小，计算出L的特征值和特征向量
- 最后，选择k个不同的特征向量对数据点聚类

&emsp;&emsp;那么如何求拉普拉斯矩阵呢？

&emsp;&emsp;将相似度矩阵`W`的每行元素相加就可以得到该顶点的度，我们定义以度为对角元素的对角矩阵称为度矩阵`D`。可以通过`A`和`D`来确定拉普拉斯矩阵。拉普拉斯矩阵分为规范和非规范两种，规范的拉普拉斯矩阵
表示为`L=D-A`，非规范的拉普拉斯矩阵表示为<img src="http://www.forkosh.com/mathtex.cgi?L=I-{D}^{-1}A">。

&emsp;&emsp;谱聚类算法的一般过程如下：

- （1）输入待聚类的数据点集以及聚类数`k`；
- （2）根据相似性度量构造数据点集的拉普拉斯矩阵`L`；
- （3）选取L的特征值和特征向量，构造特征向量空间；
- （4）使用传统方法对特征向量聚类，并对应于原始数据的聚类。

&emsp;&emsp;快速迭代算法和谱聚类算法都是将数据点嵌入到由相似矩阵推导出来的低维子空间中，然后直接或者通过`k-means`算法产生聚类结果，但是快速迭代算法有不同的地方。下面重点了解快速迭代算法的原理。

## 2 快速迭代算法的原理

&emsp;&emsp;在快速迭代算法中，我们构造另外一个矩阵<img src="http://www.forkosh.com/mathtex.cgi?W={D}^{-1}A">,同第一章做比对，我们可以知道`W`的最大特征向量就是拉普拉斯矩阵`L`的最小特征向量。
我们知道拉普拉斯矩阵有一个特性：第二小特征向量（即第二小特征值对应的特征向量）定义了图最佳划分的一个解，它可以近似最大化划分准则。更一般的，`k`个最小的特征向量所定义的子空间很适合去划分图。
因此拉普拉斯矩阵第二小、第三小直到第`k`小的特征向量可以很好的将图`W`划分为`k`个部分。

&emsp;&emsp;注意，矩阵`L`的`k`个最小特征向量也是矩阵`W`的`k`个最大特征向量。计算一个矩阵最大的特征向量可以通过一个简单的方法来求得，那就是快速迭代（即`PI`）。
`PI`是一个迭代方法，它以任意的向量<img src="http://www.forkosh.com/mathtex.cgi?{v}^{0}">作为起始，依照下面的公式循环进行更新。

<div  align="center"><img src="imgs/PIC.1.1.png" width = "120" height = "23" alt="1.1" align="center" /></div><br />

&emsp;&emsp;在上面的公式中，`c`是标准化常量，是为了避免<img src="http://www.forkosh.com/mathtex.cgi?{v}^{t}">产生过大的值，这里<img src="http://www.forkosh.com/mathtex.cgi?c=||W{v}^{t}{||}_{1}">。在大多数情况下，我们只关心第`k`（k不为1）大的特征向量，而不关注最大的特征向量。
这是因为最大的特征向量是一个常向量：因为`W`每一行的和都为1。

&emsp;&emsp;快速迭代的收敛性在文献【1】中有详细的证明，这里不再推导。

&emsp;&emsp;快速迭代算法的一般步骤如下：

<div  align="center"><img src="imgs/PIC.1.2.png" width = "480" height = "220" alt="1.2" align="center" /></div><br />

&emsp;&emsp;在上面的公式中，输入矩阵`W`根据<img src="http://www.forkosh.com/mathtex.cgi?W={D}^{-1}A">来计算。

## 3 快速迭代算法的源码实现

&emsp;&emsp;在`spark`中，文件`org.apache.spark.mllib.clustering.PowerIterationClustering`实现了快速迭代算法。我们从官方给出的例子出发来分析快速迭代算法的实现。

```scala
import org.apache.spark.mllib.clustering.{PowerIterationClustering, PowerIterationClusteringModel}
import org.apache.spark.mllib.linalg.Vectors
// 加载和切分数据
val data = sc.textFile("data/mllib/pic_data.txt")
val similarities = data.map { line =>
  val parts = line.split(' ')
  (parts(0).toLong, parts(1).toLong, parts(2).toDouble)
}
// 使用快速迭代算法将数据分为两类
val pic = new PowerIterationClustering()
  .setK(2)
  .setMaxIterations(10)
val model = pic.run(similarities)
//打印出所有的簇
model.assignments.foreach { a =>
  println(s"${a.id} -> ${a.cluster}")
}
```
&emsp;&emsp;在上面的例子中，我们知道数据分为三列，分别是起始id，目标id，以及两者的相似度，这里的`similarities`代表前面章节提到的矩阵`A`。有了数据之后，我们通过`PowerIterationClustering`的`run`方法来训练模型。
`PowerIterationClustering`类有三个参数：

- `k`：聚类数
- `maxIterations`：最大迭代数
- `initMode`：初始化模式。初始化模式分为`Random`和`Degree`两种，针对不同的模式对数据做不同的初始化操作

&emsp;&emsp;下面分步骤介绍`run`方法的实现。

- （1）标准化相似度矩阵`A`到矩阵`W`

```scala
def normalize(similarities: RDD[(Long, Long, Double)]): Graph[Double, Double] = {
    //获得所有的边
    val edges = similarities.flatMap { case (i, j, s) =>
      //相似度值必须非负
      if (s < 0.0) {
        throw new SparkException("Similarity must be nonnegative but found s($i, $j) = $s.")
      }
      if (i != j) {
        Seq(Edge(i, j, s), Edge(j, i, s))
      } else {
        None
      }
    }
    //构造图，顶点特征值默认为0
    val gA = Graph.fromEdges(edges, 0.0)
    //计算从顶点的出发的边的相似度之和，在这里称为度
    val vD = gA.aggregateMessages[Double](
      sendMsg = ctx => {
        ctx.sendToSrc(ctx.attr)
      },
      mergeMsg = _ + _,
      TripletFields.EdgeOnly)
    //计算得到W , W=A/D
    GraphImpl.fromExistingRDDs(vD, gA.edges)
      .mapTriplets(
        //gAi/vDi
        e => e.attr / math.max(e.srcAttr, MLUtils.EPSILON),
        TripletFields.Src)
  }
```
&emsp;&emsp;上面的代码首先通过边集合构造图`gA`,然后使用`aggregateMessages`计算每个顶点的度（即所有从该顶点出发的边的相似度之和），构造出`VertexRDD`。最后使用现有的`VertexRDD`和`EdgeRDD`构造图，
然后使用`mapTriplets`方法计算得到最终的图`W`。在`mapTriplets`方法中，对每一个`EdgeTriplet`，使用相似度除以出发顶点的度。




