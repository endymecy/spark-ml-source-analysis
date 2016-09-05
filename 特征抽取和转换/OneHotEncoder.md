# OneHotEncoder

&emsp;&emsp;[One-hot encoding](http://en.wikipedia.org/wiki/One-hot)将标签索引列映射为二值向量,这个向量至多有一个1值。
这个编码允许要求连续特征的算法(如逻辑回归)使用类别特征。下面是程序调用的例子。

```scala
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}

val df = spark.createDataFrame(Seq(
  (0, "a"),
  (1, "b"),
  (2, "c"),
  (3, "a"),
  (4, "a"),
  (5, "c")
)).toDF("id", "category")

val indexer = new StringIndexer()
  .setInputCol("category")
  .setOutputCol("categoryIndex")
  .fit(df)
val indexed = indexer.transform(df)

val encoder = new OneHotEncoder()
  .setInputCol("categoryIndex")
  .setOutputCol("categoryVec")
val encoded = encoder.transform(indexed)
encoded.select("id", "categoryVec").show()
```