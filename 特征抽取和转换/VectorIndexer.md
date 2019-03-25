# VectorIndexer

&emsp;&emsp;`VectorIndexer`把数据集中的类型特征进行索引。它不仅可以自动的判断哪些特征是可以类别化,也能将原有的值转换为类别索引。
通常情况下,它的过程如下:

- 1 拿到类型为`vector`的输入列和参数`maxCategories`
- 2 根据有区别的值的数量,判断哪些特征可以类别化。拥有的不同值的数量至少要为`maxCategories`的特征才能判断可以类别化。
- 3 对每一个可以类别化的特征计算基于0的类别索引。
- 4 为类别特征建立索引,将原有的特征值转换为索引。

&emsp;&emsp;索引类别特征允许诸如决策树和集合树等算法适当处理可分类的特征,提高效率。

&emsp;&emsp;在下面的例子中,我们从数据集中读取标签点,然后利用`VectorIndexer`去判断哪些特征可以被认为是可分类化的。
我们将可分类特征的值转换为索引。转换后的数据可以传递给`DecisionTreeRegressor`等可以操作分类特征的算法。

```scala
import org.apache.spark.ml.feature.VectorIndexer

val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

val indexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexed")
  .setMaxCategories(10)

val indexerModel = indexer.fit(data)

val categoricalFeatures: Set[Int] = indexerModel.categoryMaps.keys.toSet
println(s"Chose ${categoricalFeatures.size} categorical features: " +
  categoricalFeatures.mkString(", "))

// Create new column "indexed" with categorical values transformed to indices
val indexedData = indexerModel.transform(data)
indexedData.show()
```
