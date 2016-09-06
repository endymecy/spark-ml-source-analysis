# VectorSlicer

&emsp;&emsp;`VectorSlicer`是一个转换器,输入一个特征向量输出一个特征向量,它是原特征的一个子集。这在从向量列中抽取特征非常有用。
`VectorSlicer`接收一个拥有特定索引的特征列,它的输出是一个新的特征列,它的值通过输入的索引来选择。有两种类型的索引:

- 1、整数索引表示进入向量的索引,调用`setIndices()`
- 2、字符串索引表示进入向量的特征列的名称,调用`setNames()`。这种情况需要向量列拥有一个`AttributeGroup`,这是因为实现是通过属性的名字来匹配的。

&emsp;&emsp;整数和字符串都是可以使用的,并且,整数和字符串可以同时使用。至少需要选择一个特征,而且重复的特征是不被允许的。

&emsp;&emsp;输出向量首先会按照选择的索引进行排序,然后再按照选择的特征名进行排序。

## 例子

&emsp;&emsp;假设我们有下面的`DataFrame`,它的列名是`userFeatures`。

```
 userFeatures
------------------
 [0.0, 10.0, 0.5]
```
&emsp;&emsp;`userFeatures`是一个向量列,它包含三个用户特征。假设用户特征的第一列均为0,所以我们想删除它,仅仅选择后面的两列。
`VectorSlicer`通过`setIndices(1,2)`选择后面的两项,产生下面新的名为`features`的向量列。

```
 userFeatures     | features
------------------|-----------------------------
 [0.0, 10.0, 0.5] | [10.0, 0.5]
```
&emsp;&emsp;假设我们还有潜在的输入特性,如`["f1", "f2", "f3"]`,我们还可以通过`setNames("f2", "f3")`来选择。

```
 userFeatures     | features
------------------|-----------------------------
 [0.0, 10.0, 0.5] | [10.0, 0.5]
 ["f1", "f2", "f3"] | ["f2", "f3"]
```
&emsp;&emsp;下面是程序调用的例子。

```scala
import java.util.Arrays

import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, NumericAttribute}
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructType

val data = Arrays.asList(Row(Vectors.dense(-2.0, 2.3, 0.0)))

val defaultAttr = NumericAttribute.defaultAttr
val attrs = Array("f1", "f2", "f3").map(defaultAttr.withName)
val attrGroup = new AttributeGroup("userFeatures", attrs.asInstanceOf[Array[Attribute]])

val dataset = spark.createDataFrame(data, StructType(Array(attrGroup.toStructField())))

val slicer = new VectorSlicer().setInputCol("userFeatures").setOutputCol("features")

slicer.setIndices(Array(1)).setNames(Array("f3"))
// or slicer.setIndices(Array(1, 2)), or slicer.setNames(Array("f2", "f3"))

val output = slicer.transform(dataset)
println(output.select("userFeatures", "features").first())
```
