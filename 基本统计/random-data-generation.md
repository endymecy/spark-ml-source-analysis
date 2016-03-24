# 随机数生成

&emsp;&emsp;随机数生成在随机算法、性能测试中非常有用，`spark.mllib`支持生成随机的`RDD`,`RDD`的独立同分布(`iid`)的值来自于给定的分布：均匀分布、标准正太分布、泊松分布。

&emsp;&emsp;`RandomRDDs`提供工厂方法生成随机的双精度`RDD`或者向量`RDD`。下面的例子生成了一个随机的双精度`RDD`，它的值来自于标准的正太分布`N(0,1)`。

```scala
import org.apache.spark.SparkContext
import org.apache.spark.mllib.random.RandomRDDs._
val sc: SparkContext = ...
// Generate a random double RDD that contains 1 million i.i.d. values drawn from the
// standard normal distribution `N(0, 1)`, evenly distributed in 10 partitions.
val u = normalRDD(sc, 1000000L, 10)
// Apply a transform to get a random double RDD following `N(1, 4)`.
val v = u.map(x => 1.0 + 2.0 * x)
```

&emsp;&emsp;`normalRDD`的实现如下面代码所示。

```scala
def normalRDD(
      sc: SparkContext,
      size: Long,
      numPartitions: Int = 0,
      seed: Long = Utils.random.nextLong()): RDD[Double] = {
    val normal = new StandardNormalGenerator()
    randomRDD(sc, normal, size, numPartitionsOrDefault(sc, numPartitions), seed)
  }
def randomRDD[T: ClassTag](
      sc: SparkContext,
      generator: RandomDataGenerator[T],
      size: Long,
      numPartitions: Int = 0,
      seed: Long = Utils.random.nextLong()): RDD[T] = {
    new RandomRDD[T](sc, size, numPartitionsOrDefault(sc, numPartitions), generator, seed)
  }
private[mllib] class RandomRDD[T: ClassTag](sc: SparkContext,
    size: Long,
    numPartitions: Int,
    @transient private val rng: RandomDataGenerator[T],
    @transient private val seed: Long = Utils.random.nextLong) extends RDD[T](sc, Nil)
```