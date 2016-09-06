# QuantileDiscretizer

&emsp;&emsp;`QuantileDiscretizer`输入连续的特征列,输出分箱的类别特征。分箱数是通过参数`numBuckets`来指定的。
箱的范围是通过使用近似算法(见[approxQuantile ](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.sql.DataFrameStatFunctions))来得到的。
近似的精度可以通过`relativeError`参数来控制。当这个参数设置为0时,将会计算精确的分位数。箱的上边界和下边界分别是正无穷和负无穷时,
取值将会覆盖所有的实数值。

## 例子

&emsp;&emsp;假设我们有下面的`DataFrame`,它的列名是`id,hour`。

```
 id | hour
----|------
 0  | 18.0
----|------
 1  | 19.0
----|------
 2  | 8.0
----|------
 3  | 5.0
----|------
 4  | 2.2
```

&emsp;&emsp;`hour`是类型为`DoubleType`的连续特征。我们想将连续特征转换为一个分类特征。给定`numBuckets`为3,我们可以得到下面的结果。

```
id  | hour | result
----|------|------
 0  | 18.0 | 2.0
----|------|------
 1  | 19.0 | 2.0
----|------|------
 2  | 8.0  | 1.0
----|------|------
 3  | 5.0  | 1.0
----|------|------
 4  | 2.2  | 0.0
```
&emsp;&emsp;下面是代码实现的例子。

```scala
import org.apache.spark.ml.feature.QuantileDiscretizer

val data = Array((0, 18.0), (1, 19.0), (2, 8.0), (3, 5.0), (4, 2.2))
var df = spark.createDataFrame(data).toDF("id", "hour")

val discretizer = new QuantileDiscretizer()
  .setInputCol("hour")
  .setOutputCol("result")
  .setNumBuckets(3)

val result = discretizer.fit(df).transform(df)
result.show()
```
