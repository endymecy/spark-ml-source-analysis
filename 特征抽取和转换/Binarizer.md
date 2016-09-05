# Binarizer

&emsp;&emsp;`Binarization`是一个将数值特征转换为二值特征的处理过程。`threshold`参数表示决定二值化的阈值。
值大于阈值的特征二值化为1,否则二值化为0。下面是代码调用的例子。

```scala
import org.apache.spark.ml.feature.Binarizer

val data = Array((0, 0.1), (1, 0.8), (2, 0.2))
val dataFrame = spark.createDataFrame(data).toDF("label", "feature")

val binarizer: Binarizer = new Binarizer()
  .setInputCol("feature")
  .setOutputCol("binarized_feature")
  .setThreshold(0.5)

val binarizedDataFrame = binarizer.transform(dataFrame)
val binarizedFeatures = binarizedDataFrame.select("binarized_feature")
binarizedFeatures.collect().foreach(println)
```