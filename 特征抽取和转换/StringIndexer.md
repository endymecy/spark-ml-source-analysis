# StringIndexer

&emsp;&emsp;`StringIndexer`将标签列的字符串编码为标签索引。这些索引是`[0,numLabels)`,通过标签频率排序,所以频率最高的标签的索引为0。
如果输入列是数字,我们把它强转为字符串然后在编码。

## 例子

&emsp;&emsp;假设我们有下面的`DataFrame`,它的列名是`id`和`category`。

```
 id | category
----|----------
 0  | a
 1  | b
 2  | c
 3  | a
 4  | a
 5  | c
```
&emsp;&emsp;`category`是字符串列,拥有三个标签`a,b,c`。把`category`作为输入列,`categoryIndex`作为输出列,使用`StringIndexer`我们可以得到下面的结果。

```
 id | category | categoryIndex
----|----------|---------------
 0  | a        | 0.0
 1  | b        | 2.0
 2  | c        | 1.0
 3  | a        | 0.0
 4  | a        | 0.0
 5  | c        | 1.0
```
&emsp;&emsp;`a`的索引号为0是因为它的频率最高,c次之,b最后。

&emsp;&emsp;另外,`StringIndexer`处理未出现的标签的策略有两个:

- 抛出一个异常(默认情况)
- 跳过出现该标签的行

&emsp;&emsp;让我们回到上面的例子,但是这次我们重用上面的`StringIndexer`到下面的数据集。

```
 id | category
----|----------
 0  | a
 1  | b
 2  | c
 3  | d
```
&emsp;&emsp;如果我们没有为`StringIndexer`设置怎么处理未见过的标签或者设置为`error`,它将抛出异常,否则若设置为`skip`,它将得到下面的结果。

```
id  | category | categoryIndex
----|----------|---------------
 0  | a        | 0.0
 1  | b        | 2.0
 2  | c        | 1.0
```

&emsp;&emsp;下面是程序调用的例子。

```scala
import org.apache.spark.ml.feature.StringIndexer

val df = spark.createDataFrame(
  Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))
).toDF("id", "category")

val indexer = new StringIndexer()
  .setInputCol("category")
  .setOutputCol("categoryIndex")

val indexed = indexer.fit(df).transform(df)
indexed.show()
```