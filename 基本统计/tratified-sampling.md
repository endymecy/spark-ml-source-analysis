# 分层取样

&emsp;&emsp;先将总体的单位按某种特征分为若干次级总体（层），然后再从每一层内进行单纯随机抽样，组成一个样本的统计学计算方法叫做分层抽样。在`spark.mllib`中，用`key`来分层。

&emsp;&emsp;与存在于`spark.mllib`中的其它统计函数不同，分层采样方法`sampleByKey`和`sampleByKeyExact`可以在`key-value`对的`RDD`上执行。在分层采样中，可以认为`key`是一个标签，
`value`是特定的属性。例如，`key`可以是男人或者女人或者文档`id`,它相应的`value`可能是一组年龄或者是文档中的词。`sampleByKey`方法通过掷硬币的方式决定是否采样一个观察数据，
因此它需要我们传递（`pass over`）数据并且提供期望的数据大小(`size`)。`sampleByKeyExact`比每层使用`sampleByKey`随机抽样需要更多的有意义的资源，但是它能使样本大小的准确性达到了`99.99%`。

&emsp;&emsp;[sampleByKeyExact()](https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.rdd.PairRDDFunctions)允许用户准确抽取`f_k * n_k`个样本，
这里`f_k`表示期望获取键为`k`的样本的比例，`n_k`表示键为`k`的键值对的数量。下面是一个使用的例子：

```scala
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.PairRDDFunctions
val sc: SparkContext = ...
val data = ... // an RDD[(K, V)] of any key value pairs
val fractions: Map[K, Double] = ... // specify the exact fraction desired from each key
// Get an exact sample from each stratum
val approxSample = data.sampleByKey(withReplacement = false, fractions)
val exactSample = data.sampleByKeyExact(withReplacement = false, fractions)
```

&emsp;&emsp;当`withReplacement`为`true`时，采用`PoissonSampler`取样器，当`withReplacement`为`false`使，采用`BernoulliSampler`取样器。

```scala
def sampleByKey(withReplacement: Boolean,
      fractions: Map[K, Double],
      seed: Long = Utils.random.nextLong): RDD[(K, V)] = self.withScope {
    val samplingFunc = if (withReplacement) {
      StratifiedSamplingUtils.getPoissonSamplingFunction(self, fractions, false, seed)
    } else {
      StratifiedSamplingUtils.getBernoulliSamplingFunction(self, fractions, false, seed)
    }
    self.mapPartitionsWithIndex(samplingFunc, preservesPartitioning = true)
  }
def sampleByKeyExact(
      withReplacement: Boolean,
      fractions: Map[K, Double],
      seed: Long = Utils.random.nextLong): RDD[(K, V)] = self.withScope {
    val samplingFunc = if (withReplacement) {
      StratifiedSamplingUtils.getPoissonSamplingFunction(self, fractions, true, seed)
    } else {
      StratifiedSamplingUtils.getBernoulliSamplingFunction(self, fractions, true, seed)
    }
    self.mapPartitionsWithIndex(samplingFunc, preservesPartitioning = true)
  }
```
&emsp;&emsp;下面我们分别来看`sampleByKey`和`sampleByKeyExact`的实现。

## 1 `sampleByKey`的实现

&emsp;&emsp;当我们需要不重复抽样时，我们需要用泊松抽样器来抽样。当需要重复抽样时，用伯努利抽样器抽样。`sampleByKey`的实现比较简单，它就是统一的随机抽样。

### 1.1 泊松抽样器

&emsp;&emsp;我们首先看泊松抽样器的实现。

```scala
def getPoissonSamplingFunction[K: ClassTag, V: ClassTag](rdd: RDD[(K, V)],
      fractions: Map[K, Double],
      exact: Boolean,
      seed: Long): (Int, Iterator[(K, V)]) => Iterator[(K, V)] = {
      (idx: Int, iter: Iterator[(K, V)]) => {
              //初始化随机生成器
              val rng = new RandomDataGenerator()
              rng.reSeed(seed + idx)
              iter.flatMap { item =>
                //获得下一个泊松值
                val count = rng.nextPoisson(fractions(item._1))
                if (count == 0) {
                  Iterator.empty
                } else {
                  Iterator.fill(count)(item)
                }
              }
            }
}
```
&emsp;&emsp;`getPoissonSamplingFunction`返回的是一个函数，传递给`mapPartitionsWithIndex`处理每个分区的数据。这里`RandomDataGenerator`是一个随机生成器，它用于同时生成均匀值(`uniform values`)和泊松值(`Poisson values`)。

### 1.2 伯努利抽样器

```scala
def getBernoulliSamplingFunction[K, V](rdd: RDD[(K, V)],
      fractions: Map[K, Double],
      exact: Boolean,
      seed: Long): (Int, Iterator[(K, V)]) => Iterator[(K, V)] = {
    var samplingRateByKey = fractions
    (idx: Int, iter: Iterator[(K, V)]) => {
      //初始化随机生成器
      val rng = new RandomDataGenerator()
      rng.reSeed(seed + idx)
      // Must use the same invoke pattern on the rng as in getSeqOp for without replacement
      // in order to generate the same sequence of random numbers when creating the sample
      iter.filter(t => rng.nextUniform() < samplingRateByKey(t._1))
    }
  }
```

## 2 `sampleByKeyExact`的实现

&emsp;&emsp;`sampleByKeyExact`获取更准确的抽样结果，它的实现也分为两种情况，重复抽样和不重复抽样。前者使用泊松抽样器，后者使用伯努利抽样器。

### 2.1 泊松抽样器

```scala
val counts = Some(rdd.countByKey())
//计算立即接受的样本数量，并且为每层生成候选名单
val finalResult = getAcceptanceResults(rdd, true, fractions, counts, seed)
//决定接受样本的阈值，生成准确的样本大小
val thresholdByKey = computeThresholdByKey(finalResult, fractions)
(idx: Int, iter: Iterator[(K, V)]) => {
     val rng = new RandomDataGenerator()
     rng.reSeed(seed + idx)
     iter.flatMap { item =>
          val key = item._1
          val acceptBound = finalResult(key).acceptBound
          // Must use the same invoke pattern on the rng as in getSeqOp for with replacement
          // in order to generate the same sequence of random numbers when creating the sample
          val copiesAccepted = if (acceptBound == 0) 0L else rng.nextPoisson(acceptBound)
          //候选名单
          val copiesWaitlisted = rng.nextPoisson(finalResult(key).waitListBound)
          val copiesInSample = copiesAccepted +
            (0 until copiesWaitlisted).count(i => rng.nextUniform() < thresholdByKey(key))
          if (copiesInSample > 0) {
            Iterator.fill(copiesInSample.toInt)(item)
          } else {
            Iterator.empty
          }
     }
}
```

### 2.2 伯努利抽样

```scala
def getBernoulliSamplingFunction[K, V](rdd: RDD[(K, V)],
      fractions: Map[K, Double],
      exact: Boolean,
      seed: Long): (Int, Iterator[(K, V)]) => Iterator[(K, V)] = {
    var samplingRateByKey = fractions
    //计算立即接受的样本数量，并且为每层生成候选名单
    val finalResult = getAcceptanceResults(rdd, false, fractions, None, seed)
    //决定接受样本的阈值，生成准确的样本大小
    samplingRateByKey = computeThresholdByKey(finalResult, fractions)
    (idx: Int, iter: Iterator[(K, V)]) => {
      val rng = new RandomDataGenerator()
      rng.reSeed(seed + idx)
      // Must use the same invoke pattern on the rng as in getSeqOp for without replacement
      // in order to generate the same sequence of random numbers when creating the sample
      iter.filter(t => rng.nextUniform() < samplingRateByKey(t._1))
    }
  }
```