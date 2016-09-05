# PolynomialExpansion(多元展开)

&emsp;&emsp;[Polynomial expansion](http://en.wikipedia.org/wiki/Polynomial_expansion)是一个将特征展开到多元空间的处理过程。
它通过`n-degree`结合原始的维度来定义。比如设置`degree`为2就可以将`(x, y)`转化为`(x, x x, y, x y, y y)`。`PolynomialExpansion`提供了这个功能。
下面的例子展示了如何将特征展开为一个`3-degree`多项式空间。

```scala
import org.apache.spark.ml.feature.PolynomialExpansion
import org.apache.spark.ml.linalg.Vectors

val data = Array(
  Vectors.dense(-2.0, 2.3),
  Vectors.dense(0.0, 0.0),
  Vectors.dense(0.6, -1.1)
)
val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
val polynomialExpansion = new PolynomialExpansion()
  .setInputCol("features")
  .setOutputCol("polyFeatures")
  .setDegree(3)
val polyDF = polynomialExpansion.transform(df)
polyDF.select("polyFeatures").take(3).foreach(println)
```