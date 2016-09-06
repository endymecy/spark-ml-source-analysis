# SQLTransformer

&emsp;&emsp;`SQLTransformer`实现了一种转换,这个转换通过`SQl`语句来定义。目前我们仅仅支持的`SQL`语法是像`SELECT ... FROM __THIS__ ...`的形式。
这里`__THIS__`表示输入数据集相关的表。例如,`SQLTransformer`支持的语句如下:

- `SELECT a, a + b AS a_b FROM __THIS__`
- `SELECT a, SQRT(b) AS b_sqrt FROM __THIS__ where a > 5`
- `SELECT a, b, SUM(c) AS c_sum FROM __THIS__ GROUP BY a, b`

## 例子

&emsp;&emsp;假设我们拥有下面的`DataFrame`,它的列名是`id,v1,v2`。

```
id  |  v1 |  v2
----|-----|-----
 0  | 1.0 | 3.0
 2  | 2.0 | 5.0
```
&emsp;&emsp;下面是语句`SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__`的输出结果。

```
 id |  v1 |  v2 |  v3 |  v4
----|-----|-----|-----|-----
 0  | 1.0 | 3.0 | 4.0 | 3.0
 2  | 2.0 | 5.0 | 7.0 |10.0
```
&emsp;&emsp;下面是程序调用的例子。

```scala
import org.apache.spark.ml.feature.SQLTransformer

val df = spark.createDataFrame(
  Seq((0, 1.0, 3.0), (2, 2.0, 5.0))).toDF("id", "v1", "v2")

val sqlTrans = new SQLTransformer().setStatement(
  "SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__")

sqlTrans.transform(df).show()
```
