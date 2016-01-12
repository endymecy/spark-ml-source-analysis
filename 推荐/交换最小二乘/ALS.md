# 1 什么是ALS

&emsp;&emsp;ALS 是交替最小二乘（`alternating least squares`）的简称。在机器学习中，`ALS`特指使用交替最小二乘求解的一个协同推荐算法。它通过观察到的所有用户给商品的打分，来推断每个用户的喜好并向用户推荐适合的商品。举个例子，我们看下面一个`8*8`的用户打分矩阵。

<div  align="center"><img src="imgs/ALS.1.1.png" width = "450" height = "300" alt="8*8打分" align="center" /></div>

&emsp;&emsp;&emsp;这个矩阵的每一行代表一个用户`（u1,u2,…,u8）`、每一列代表一个商品`（v1,v2,…,v8）`、用户的打分为`1-9`分。这个矩阵只显示了观察到的打分，我们需要推测没有观察到的打分。比如`（u6，v5）`打分多少？如果以数独的方式来解决这个问题，可以得到唯一的结果。
因为数独的规则很强，每添加一条规则，就让整个系统的自由度下降一个量级。当我们满足所有的规则时，整个系统的自由度就降为`1`了，也就得出了唯一的结果。对于上面的打分矩阵，如果我们不添加任何条件的话，也即打分之间是相互独立的，我们就没法得到`（u6，v5）`的打分。
所以在这个用户打分矩阵的基础上，我们需要提出一个限制其自由度的合理假设，使得我们可以通过观察已有打分来猜测未知打分。

&emsp;&emsp;`ALS`的核心就是这样一个假设：打分矩阵是近似低秩的。换句话说，就是一个`m*n`的打分矩阵可以由分解的两个小矩阵`U（m*k）`和`V（k*n）`的乘积来近似，即<img src="http://www.forkosh.com/mathtex.cgi? A=U{A}^{T},k <= m,n">。这就是`ALS`的矩阵分解方法。这样我们把系统的自由度从`O(mn)`降到了`O((m+n)k)`。

&emsp;&emsp;那么`ALS`的低秩假设为什么是合理的呢？我们描述一个人的喜好经常是在一个抽象的低维空间上进行的，并不需要一一列出他喜好的事物。例如，我喜好看侦探影片，可能代表我喜欢《神探夏洛特》、《神探狄仁杰》等。这些影片都符合我对自己喜好的描述，也就是说他们在这个抽象的低维空间的投影和我的喜好相似。
再抽象一些来描述这个问题，我们把某个人的喜好映射到了低维向量`ui`上，同时将某个影片的特征映射到了维度相同的向量`vj`上，那么这个人和这个影片的相似度就可以表述成这两个向量之间的内积<img src="http://www.forkosh.com/mathtex.cgi?{{u}_{i}}^{T}{v}_{j}">。
我们把打分理解成相似度，那么打分矩阵A就可以由用户喜好矩阵和产品特征矩阵的乘积<img src="http://www.forkosh.com/mathtex.cgi?{U}{V}^{T}">来近似了。

&emsp;&emsp;低维空间的选取是一个问题。这个低维空间要能够很好的区分事物，那么就需要一个明确的可量化目标，这就是重构误差。在`ALS`中我们使用F范数来量化重构误差，就是每个元素重构误差的平方和。这里存在一个问题，我们只观察到部分打分，`A`中的大量未知元是我们想推断的，所以这个重构误差是包含未知数的。
解决方案很简单：只计算已知打分的重构误差。

<div  align="center"><img src="imgs/math.1.1.png" width = "125" height = "25" alt="重构误差" align="center" /></div>


&emsp;&emsp;后面的章节我们将从原理上讲解spark中实现的ALS模型。

# 2 spark中ALS的实现原理

&emsp;&emsp;`Spark`利用交换最小二乘解决矩阵分解问题分两种情况：数据集是显式反馈和数据集是隐式反馈。由于隐式反馈算法的原理是在显示反馈算法原理的基础上作的修改，所以我们在此只会具体讲解数据集为隐式反馈的算法。
算法实现所依据的文献为:[Collaborative Filtering for Implicit Feedback Datasets](papers/Collaborative Filtering for Implicit Feedback Datasets.pdf)。

## 2.1 介绍

&emsp;&emsp;从广义上讲，推荐系统基于两种不同的策略：基于内容的方法和基于协同过滤的方法。`Spark`中使用协同过滤的方式。协同过滤分析用户以及用户相关的产品的相关性，用以识别新的用户-产品相关性。协同过滤系统需要的唯一信息是用户过去的行为信息，比如对产品的评价信息。协同过滤是领域无关的，所以它可以方便解决基于内容方法难以解决的许多问题。

&emsp;&emsp;推荐系统依赖不同类型的输入数据，最方便的是高质量的显式反馈数据，它们包含用户对感兴趣商品明确的评价。例如，`Netflix`收集的用户对电影评价的星星等级数据。但是显式反馈数据不一定总是找得到，因此推荐系统可以从更丰富的隐式反馈信息中推测用户的偏好。
隐式反馈类型包括购买历史、浏览历史、搜索模式甚至鼠标动作。例如，购买同一个作者许多书的用户可能喜欢这个作者。

&emsp;&emsp;许多研究都集中在处理显式反馈，然而在很多应用场景下，应用程序重点关注隐式反馈数据。因为可能用户不愿意评价商品或者系统性质我们不能收集显式反馈数据。在隐式模型中，一旦用户允许收集可用的数据，在客户端并不需要额外的显示数据。文献中的系统避免主动地向用户收集显式反馈信息，所以系统紧紧依靠隐式信息。

&emsp;&emsp;了解隐式反馈的特质非常重要，因为这些特质使我们避免了直接调用基于显示反馈的算法。最主要的特质有如下几种：

- （1）	没有负反馈。通过观察用户行为，我们可以推测那个商品他可能喜欢，然后购买，但是我们很难推测哪个商品用户不喜欢。这在显式反馈算法中并不存在，因为用户明确告诉了我们哪些他喜欢哪些他不喜欢。
- （2）	隐式反馈是内在的噪音。虽然我们拼命的追踪用户行为，但是我们仅仅只是猜测他们的偏好和真实动机。例如，我们可能知道一个人的购买行为，但是这并不能表明一个积极的观点，这个商品可能作为礼物被购买或者用户并不喜欢它。
- （3）	显示反馈的数值值表示偏好（`preference`），隐式回馈的数值值表示信任（`confidence`）。基于显示反馈的系统用星星等级让用户表达他们的喜好层度，例如一颗星表示很不喜欢，五颗星表示非常喜欢。基于隐式反馈的数值值描述的是动作的频率，例如用户购买特定商品的次数。一个较大的值并不能表明更多的偏爱。但是这个值是有用的，它描述了在一个特定观察中的信任度。
一个发生一次的事件可能对用户偏爱没有用，但是一个周期性事件更可能反映一个用户的选择。
- （4）	评价隐式反馈推荐系统需要合适的手段。

## 2.2 显式反馈模型

&emsp;&emsp;潜在因素模型由一个针对协同过滤的交替方法组成，它以一个更加全面的方式发现潜在特征来解释观察的`ratings`数据。我们关注的模型由奇异值分解（`SVD`）推演而来。一个典型的模型将每个用户`u`（包含一个用户-因素向量`ui`）和每个商品`v`（包含一个用户-因素向量`vj`）联系起来。
预测通过内积<img src="http://www.forkosh.com/mathtex.cgi?{r}_{ij}={{u}_{i}}^{T}{v}_{j}">来实现。另一个需要关注的地方是参数估计。许多当前的工作都应用到了显式反馈数据集中，这些模型仅仅基于观察到的`rating`数据直接建模，同时通过一个适当的正则化来避免过拟合。公式如下：

<div  align="center"><img src="imgs/math.2.1.png" width = "425" height = "50" alt="重构误差" align="center" /></div>

&emsp;&emsp;在公式(2.1)中，λ是正则化的参数。就这样，我们用最小化重构误差来解决协同推荐问题。我们也成功将推荐问题转换为了最优化问题。

## 2.3 隐式反馈模型

&emsp;&emsp;在显式反馈的基础上，我们需要做一些改动得到我们的隐式反馈模型。首先，我们需要形式化由<img src="http://www.forkosh.com/mathtex.cgi?{r}_{ij}">变量衡量的信任度的概念。我们引入了一组二元变量<img src="http://www.forkosh.com/mathtex.cgi?{p}_{ij}">，它表示用户u对商品v的偏好。<img src="http://www.forkosh.com/mathtex.cgi?{p}_{ij}">的公式如下：

<div  align="center"><img src="imgs/math.2.2.png" width = "325" height = "50" alt="p形式" align="center" /></div>

&emsp;&emsp;换句话说，如果用户购买了商品，我们认为用户喜欢该商品，否则我们认为用户不喜欢该商品。然而我们的信念（`beliefs`）与变化的信任（`confidence`）等级息息相关。首先，很自然的，<img src="http://www.forkosh.com/mathtex.cgi?{p}_{ij}">的值为0和低信任有关。用户对一个商品没有得到一个正的偏好可能源于多方面的原因，并不一定是不喜欢该商品。例如，用户可能并不知道该商品的存在。
另外，用户购买一个商品也并不一定是用户喜欢它。因此我们需要一个新的信任等级来显示用户偏爱某个商品。一般情况下，<img src="http://www.forkosh.com/mathtex.cgi?{r}_{ij}">越大，越能暗示用户喜欢某个商品。因此，我们引入了一组变量<img src="http://www.forkosh.com/mathtex.cgi?{c}_{ij}">，它衡量了我们观察到<img src="http://www.forkosh.com/mathtex.cgi?{p}_{ij}">的信任度。<img src="http://www.forkosh.com/mathtex.cgi?{c}_{ij}">一个合理的选择如下所示：

<div  align="center"><img src="imgs/math.2.3.png" width = "280" height = "25" alt="信任度" align="center" /></div>

&emsp;&emsp;按照这种方式，我们存在最小限度的信任度，并且随着我们观察到的正偏向的证据越来越多，信任度也会越来越大。

&emsp;&emsp;我们的目的是找到用户向量`ui`以及商品向量`vj`来表明用户偏好。这些向量分别是用户因素向量和商品因素向量。本质上，这些向量将用户和商品映射到一个公用的隐式因素空间，从而使它们可以直接比较。这和用于显式数据集的矩阵分解技术类似，但是包含两点不一样的地方：
（1）我们需要考虑不同的信任度，（2）最优化需要考虑所有可能的u，v对，而不仅仅是和观察数据相关的u，v对。因此，通过最小化下面的损失函数来计算相关因素（`factors`）。

<div  align="center"><img src="imgs/math.2.4.png" width = "450" height = "50" alt="信任度" align="center" /></div>

## 2.4 求解最小化损失函数

&emsp;&emsp;考虑到损失函数包含`m*n`个元素，`m`是用户的数量，`n`是商品的数量。一般情况下，`m*n`可以到达几百亿。这么多的元素应该避免使用随机梯度下降法来求解，因此，spark选择使用交替最优化方式求解。

&emsp;&emsp;公式（2.1）和公式（2.4）是非凸函数，无法求解最优解。但是，固定公式中的用户-特征向量或者商品-特征向量，公式就会变成二次方程，可以求出全局的极小值。这样就产生了交替最小二乘的优化过程：我们交替的重新计算用户-特征向量和商品-特征向量，每一步都保证降低损失函数的值。
交替最小二乘法的处理过程如下所示：

<div  align="center"><img src="imgs/ALS.2.1.png" width = "375" height = "90" alt="交替最小二乘法处理流程" align="center" /></div>

# 3 ALS在spark中的实现

&emsp;&emsp;在`spark`的源代码中，`ALS`算法实现于`org.apache.spark.ml.recommendation.ALS.scala`文件中。我们以官方文档中的例子为起点，来分析`ALS`算法的分布式实现。下面是官方的例子：

```scala
//处理训练数据
val data = sc.textFile("data/mllib/als/test.data")
val ratings = data.map(_.split(',') match { case Array(user, item, rate) =>
  Rating(user.toInt, item.toInt, rate.toDouble)
})
// 使用ALS训练推荐模型
val rank = 10
val numIterations = 10
val model = ALS.train(ratings, rank, numIterations, 0.01)
```

&emsp;&emsp;从代码中我们知道，训练模型用到了`ALS.scala`文件中的`train`方法，下面我们将详细介绍`train`方法的实现。在此之前，我们先了解一下`train`方法的参数表示的含义。

```scala
def train( 
    ratings: RDD[Rating[ID]],  //训练数据
    rank: Int = 10,   //隐含特征数
    numUserBlocks: Int = 10, //分区数
    numItemBlocks: Int = 10,
    maxIter: Int = 10,   //迭代次数
    regParam: Double = 1.0,
    implicitPrefs: Boolean = false,
    alpha: Double = 1.0,
    nonnegative: Boolean = false,
    intermediateRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
    finalRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
    checkpointInterval: Int = 10,
    seed: Long = 0L)
```

&emsp;&emsp;在这段代码中，`ratings`指用户提供的训练数据，它包括用户id集、商品id集和打分集。`rank`表示隐含因素的数量，也即特征的数量。`numUserBlocks`和`numItemBlocks`分别指用户和商品的块数量，即分区数量。`maxIter`表示迭代次数。`regParam`表示最小二乘法中`lambda`值的大小。
`implicitPrefs`表示我们的训练数据是否是隐式反馈数据。`Nonnegative`表示求解的最小二乘的值是否是非负,根据`Nonnegative`的值的不同，`spark`使用了不同的矩阵分解方法。

&emsp;&emsp;下面我们分步骤分析`train`方法的处理流程。

- **(1)初始化`ALSPartitioner`和`LocalIndexEncoder`**。

&emsp;&emsp;`ALSPartitioner`实现了基于`hash`的分区，它根据用户或者商品id的`hash`值来进行分区。`LocalIndexEncoder`对`（blockid，localindex）`即`（分区id，分区内索引）`进行编码，并将其转换为一个整数，这个整数在高位存分区ID，在低位存对应分区的索引，在空间上尽量做到了不浪费。
同时也可以根据这个转换的整数分别获得`blockid`和`localindex`。这两个对象在后续的代码中会用到。

```scala
val userPart = new ALSPartitioner(numUserBlocks)
val itemPart = new ALSPartitioner(numItemBlocks)
val userLocalIndexEncoder = new LocalIndexEncoder(userPart.numPartitions)
val itemLocalIndexEncoder = new LocalIndexEncoder(itemPart.numPartitions)
```

- **(2)根据`nonnegative`参数选择解决矩阵分解的方法**。

&emsp;&emsp;如果需要解的值为非负,即`nonnegative`为`true`，那么用非负正则化最小二乘来解，如果没有这个限制，用乔里斯基分解来解。这两个算法我们在最优化模块作了详细讲解。

```scala
val solver = if (nonnegative) new NNLSSolver else new CholeskySolver
```

- **(3)将`ratings`数据转换为分区的格式**。

&emsp;&emsp;将`ratings`数据转换为分区的形式，即`（（用户分区id，商品分区id），分区数据集blocks））`的形式，并缓存到内存中。其中分区id的计算是通过`ALSPartitioner`的`getPartitions`方法获得的，分区数据集由`RatingBlock`组成，
它表示`（用户分区id，商品分区id ）`对所对应的用户id集，商品id集，以及打分集，即`（用户id集，商品id集，打分集）`。

```scala
val blockRatings = partitionRatings(ratings, userPart, itemPart)
  .persist(intermediateRDDStorageLevel)
  
//以下是partitionRatings的实现
  //默认是10*10
  val numPartitions = srcPart.numPartitions * dstPart.numPartitions
  ratings.mapPartitions { iter =>
      val builders = Array.fill(numPartitions)(new RatingBlockBuilder[ID])
      iter.flatMap { r =>
        val srcBlockId = srcPart.getPartition(r.user)
        val dstBlockId = dstPart.getPartition(r.item)
        //当前builder的索引位置
        val idx = srcBlockId + srcPart.numPartitions * dstBlockId
        val builder = builders(idx)
        builder.add(r)
        //如果某个builder的数量大于2048，那么构建一个分区
        if (builder.size >= 2048) { // 2048 * (3 * 4) = 24k
          builders(idx) = new RatingBlockBuilder
          //单元素集合
          Iterator.single(((srcBlockId, dstBlockId), builder.build()))
        } else {
          Iterator.empty
        }
      } ++ {
        builders.view.zipWithIndex.filter(_._1.size > 0).map { case (block, idx) =>
          //用户分区id
          val srcBlockId = idx % srcPart.numPartitions
          //商品分区id
          val dstBlockId = idx / srcPart.numPartitions
          ((srcBlockId, dstBlockId), block.build())
        }
      }
    }.groupByKey().mapValues { blocks =>
      val builder = new RatingBlockBuilder[ID]
      blocks.foreach(builder.merge)
      builder.build()
    }.setName("ratingBlocks")
  }
```

- **（4）获取`inblocks`和`outblocks`数据**。

&emsp;&emsp;获取`inblocks`和`outblocks`数据是数据处理的重点。我们知道，通信复杂度是分布式实现一个算法时要重点考虑的问题，不同的实现可能会对性能产生很大的影响。我们假设最坏的情况：即求解商品需要的所有用户特征都需要从其它节点获得。
如下图3.1所示，求解`v1`需要获得`u1`,`u2`，求解`v2`需要获得`u1`,`u2`,`u3`等，在这种假设下，每步迭代所需的交换数据量是`O(m*rank)`，其中`m`表示所有观察到的打分集大小，`rank`表示特征数量。

<div  align="center"><img src="imgs/ALS.3.1.png" width = "520" height = "260" alt="例子1" align="center" /></div>