# RFormula

&emsp;&emsp;`RFormula`通过一个[R model formula](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/formula.html)选择一个特定的列。
目前我们支持`R`算子的一个受限的子集,包括`~`,`.`,`:`,`+`,`-`。这些基本的算子是:

- `~` 分开`target`和`terms`
- `+` 连接`term`,`+ 0`表示删除截距(`intercept`)
- `-` 删除`term`,`- 1`表示删除截距
- `:` 交集
- `.` 除了`target`之外的所有列

&emsp;&emsp;假设`a`和`b`是`double`列,我们用下面简单的例子来证明`RFormula`的有效性。

- `y ~ a + b` 表示模型 `y ~ w0 + w1 * a + w2 * b`,其中`w0`是截距,`w1`和`w2`是系数
- `y ~ a + b + a:b - 1`表示模型`y ~ w1 * a + w2 * b + w3 * a * b`,其中`w1`,`w2`,`w3`是系数

&emsp;&emsp;`RFormula`产生一个特征向量列和一个`double`或`string`类型的标签列。比如在线性回归中使用`R`中的公式时,
字符串输入列是`one-hot`编码,数值列强制转换为`double`类型。如果标签列是字符串类型,它将使用`StringIndexer`转换为`double`
类型。如果`DataFrame`中不存在标签列,输出的标签列将通过公式中指定的返回变量来创建。

## 例子

&emsp;&emsp;假设我们有一个`DataFrame`,它的列名是`id`, `country`, `hour`和`clicked`。

```
id | country | hour | clicked
---|---------|------|---------
 7 | "US"    | 18   | 1.0
 8 | "CA"    | 12   | 0.0
 9 | "NZ"    | 15   | 0.0
```
&emsp;&emsp;如果我们用`clicked ~ country + hour`(基于`country`和`hour`来预测`clicked`)来作用于`RFormula`,将会得到下面的结果。

```
id | country | hour | clicked | features         | label
---|---------|------|---------|------------------|-------
 7 | "US"    | 18   | 1.0     | [0.0, 0.0, 18.0] | 1.0
 8 | "CA"    | 12   | 0.0     | [0.0, 1.0, 12.0] | 0.0
 9 | "NZ"    | 15   | 0.0     | [1.0, 0.0, 15.0] | 0.0
```
&emsp;&emsp;下面是代码调用的例子。

```scala
import org.apache.spark.ml.feature.RFormula

val dataset = spark.createDataFrame(Seq(
  (7, "US", 18, 1.0),
  (8, "CA", 12, 0.0),
  (9, "NZ", 15, 0.0)
)).toDF("id", "country", "hour", "clicked")
val formula = new RFormula()
  .setFormula("clicked ~ country + hour")
  .setFeaturesCol("features")
  .setLabelCol("label")
val output = formula.fit(dataset).transform(dataset)
output.select("features", "label").show()
```