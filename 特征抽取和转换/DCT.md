# Discrete Cosine Transform (DCT)

&emsp;&emsp;[Discrete Cosine Transform](https://en.wikipedia.org/wiki/Discrete_cosine_transform)将一个在时间域(`time domain`)内长度为`N`的实值序列转换为另外一个
在频率域(`frequency domain`)内的长度为`N`的实值序列。下面是程序调用的例子。

```scala
import org.apache.spark.ml.feature.DCT
import org.apache.spark.ml.linalg.Vectors

val data = Seq(
  Vectors.dense(0.0, 1.0, -2.0, 3.0),
  Vectors.dense(-1.0, 2.0, 4.0, -7.0),
  Vectors.dense(14.0, -2.0, -5.0, 1.0))

val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

val dct = new DCT()
  .setInputCol("features")
  .setOutputCol("featuresDCT")
  .setInverse(false)

val dctDf = dct.transform(df)
dctDf.select("featuresDCT").show(3)
```