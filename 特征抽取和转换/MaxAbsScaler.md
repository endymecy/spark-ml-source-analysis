# MaxAbsScaler

&emsp;&emsp;`MaxAbsScaler`转换由向量列组成的数据集,将每个特征调整到`[-1,1]`的范围,它通过每个特征内的最大绝对值来划分。
它不会移动和聚集数据,因此不会破坏任何的稀疏性。

`MaxAbsScaler`计算数据集上的统计数据,生成`MaxAbsScalerModel`,然后使用生成的模型分别的转换特征到范围`[-1,1]`。下面是程序调用的例子。

```scala
import org.apache.spark.ml.feature.MaxAbsScaler

val dataFrame = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
val scaler = new MaxAbsScaler()
  .setInputCol("features")
  .setOutputCol("scaledFeatures")

// Compute summary statistics and generate MaxAbsScalerModel
val scalerModel = scaler.fit(dataFrame)

// rescale each feature to range [-1, 1]
val scaledData = scalerModel.transform(dataFrame)
scaledData.show()
```