# Bucketizer

&emsp;&emsp;`Bucketizer`将连续的特征列转换成特征桶(`buckets`)列。这些桶由用户指定。它拥有一个`splits`参数。

- `splits`:如果有`n+1`个`splits`,那么将有`n`个桶。桶将由`split x`和`split y`共同确定,它的值范围为`[x,y)`,如果是最后
一个桶,范围将是`[x,y]`。`splits`应该严格递增。负无穷和正无穷必须明确的提供用来覆盖所有的双精度值,否则,超出`splits`的值将会被
认为是一个错误。`splits`的两个例子是`Array(Double.NegativeInfinity, 0.0, 1.0, Double.PositiveInfinity)` 和 `Array(0.0, 1.0, 2.0)`。

&emsp;&emsp;注意,如果你并不知道目标列的上界和下界,你应该添加`Double.NegativeInfinity`和`Double.PositiveInfinity`作为边界从而防止潜在的
超过边界的异常。下面是程序调用的例子。

```scala
import org.apache.spark.ml.feature.Bucketizer

val splits = Array(Double.NegativeInfinity, -0.5, 0.0, 0.5, Double.PositiveInfinity)

val data = Array(-0.5, -0.3, 0.0, 0.2)
val dataFrame = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

val bucketizer = new Bucketizer()
  .setInputCol("features")
  .setOutputCol("bucketedFeatures")
  .setSplits(splits)

// Transform original data into its bucket index.
val bucketedData = bucketizer.transform(dataFrame)
bucketedData.show()
```