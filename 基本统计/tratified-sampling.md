# 分层取样

&emsp;&emsp;先将总体的单位按某种特征分为若干次级总体（层），然后再从每一层内进行单纯随机抽样，组成一个样本的统计学计算方法叫做分层抽样。

&emsp;&emsp;与存在于`spark.mllib`中的其它统计函数不同，分层采样方法`sampleByKey`和`sampleByKeyExact`可以在`key-value`对的`RDD`上执行。在分层采样中，可以认为`key`是一个标签，
`value`是特定的属性。例如，`key`可以是男人或者女人或者文档`id`,它相应的`value`可能是一组年龄或者是文档中的词。`sampleByKey`方法通过掷硬币的方式决定是否采样一个观察数据，
因此它需要我们忽视（`pass over`）数据本身而只提供期望的数据大小。`sampleByKeyExact`比每层使用`sampleByKey`随机抽样需要更多的有意义的资源，但是它能使样本大小的准确性达到了`99.99%`。

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

当`withReplacement`为`true`时，采用`PoissonSampler`取样器，当`withReplacement`为`false`使，采用`BernoulliSampler`取样器。

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
