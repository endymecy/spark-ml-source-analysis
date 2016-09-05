# IndexToString

&emsp;&emsp;与`StringIndexer`相对的是,`IndexToString`将标签索引列映射回原来的字符串标签。一个通用的使用案例是使用
`StringIndexer`将标签转换为索引,然后通过索引训练模型,最后通过`IndexToString`将预测的标签索引恢复成字符串标签。

## 例子

&emsp;&emsp;假设我们有下面的`DataFrame`,它的列名为`id`和`categoryIndex`。

```
 id | categoryIndex
----|---------------
 0  | 0.0
 1  | 2.0
 2  | 1.0
 3  | 0.0
 4  | 0.0
 5  | 1.0
```
&emsp;&emsp;把`categoryIndex`作为输入列,`originalCategory`作为输出列,使用`IndexToString`我们可以恢复原来的标签。

```
id  | categoryIndex | originalCategory
----|---------------|-----------------
 0  | 0.0           | a
 1  | 2.0           | b
 2  | 1.0           | c
 3  | 0.0           | a
 4  | 0.0           | a
 5  | 1.0           | c
```
&emsp;&emsp;下面是程序调用的例子。

```scala
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}

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

val converter = new IndexToString()
  .setInputCol("categoryIndex")
  .setOutputCol("originalCategory")

val converted = converter.transform(indexed)
converted.select("id", "originalCategory").show()
```