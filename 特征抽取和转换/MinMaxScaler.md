# MinMaxScaler

&emsp;&emsp;`MinMaxScaler`转换由向量行组成的数据集,将每个特征调整到一个特定的范围(通常是`[0,1]`)。它有下面两个参数:

- `min`:默认是0。转换的下界,被所有的特征共享。
- `max`:默认是1。转换的上界,被所有特征共享。

&emsp;&emsp;`MinMaxScaler`计算数据集上的概要统计数据,产生一个`MinMaxScalerModel`。然后就可以用这个模型单独的转换每个特征到特定的范围。
特征`E`被转换后的值可以用下面的公式计算:

$$\frac{e_{i} - E_{min}}{E_{max} - E_{min}} * (max - min) + min$$

&emsp;&emsp;对于`E_{max} == E_{min}`的情况,`Rescaled(e_i) = 0.5 * (max + min)`。

&emsp;&emsp;注意,由于0值有可能转换成非0的值,所以转换的输出为`DenseVector`,即使输入为稀疏的数据也一样。下面的例子展示了如何将特征转换到`[0,1]`。

```scala
import org.apache.spark.ml.feature.MinMaxScaler

val dataFrame = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

val scaler = new MinMaxScaler()
  .setInputCol("features")
  .setOutputCol("scaledFeatures")

// Compute summary statistics and generate MinMaxScalerModel
val scalerModel = scaler.fit(dataFrame)

// rescale each feature to range [min, max].
val scaledData = scalerModel.transform(dataFrame)
scaledData.show()
```