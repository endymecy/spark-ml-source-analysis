# VectorAssembler

&emsp;&emsp;`VectorAssembler`是一个转换器,它可以将给定的多列转换为一个向量列。合并原始特征与通过不同的转换器转换而来的特征,从而训练机器学习模型,
`VectorAssembler`是非常有用的。`VectorAssembler`允许这些类型:所有的数值类型,`boolean`类型以及`vector`类型。

## 例子

&emsp;&emsp;假设我们有下面的`DataFrame`,它的列名分别是`id, hour, mobile, userFeatures, clicked`。

```
id  | hour | mobile | userFeatures     | clicked
----|------|--------|------------------|---------
 0  | 18   | 1.0    | [0.0, 10.0, 0.5] | 1.0
```

&emsp;&emsp;`userFeatures`是一个向量列,包含三个用户特征。我们想合并`hour`, `mobile`和`userFeatures`到一个名为`features`的特征列。
通过转换之后,我们可以得到下面的结果。

```
id  | hour | mobile | userFeatures     | clicked | features
----|------|--------|------------------|---------|-----------------------------
 0  | 18   | 1.0    | [0.0, 10.0, 0.5] | 1.0     | [18.0, 1.0, 0.0, 10.0, 0.5]
```

&emsp;&emsp;下面是程序调用的例子。

```scala
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

val dataset = spark.createDataFrame(
  Seq((0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0))
).toDF("id", "hour", "mobile", "userFeatures", "clicked")

val assembler = new VectorAssembler()
  .setInputCols(Array("hour", "mobile", "userFeatures"))
  .setOutputCol("features")

val output = assembler.transform(dataset)
println(output.select("features", "clicked").first())
```