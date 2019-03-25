# 正则化

&emsp;&emsp;正则化器缩放单个样本让其拥有单位$L^{p}$范数。这是文本分类和聚类常用的操作。例如，两个$L^{2}$正则化的`TFIDF`向量的点乘就是两个向量的`cosine`相似度。

&emsp;&emsp;`Normalizer`实现` VectorTransformer`，将一个向量正则化为转换的向量，或者将一个`RDD`规则化为另一个`RDD`。下面是一个正则化的例子。

```scala
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.feature.Normalizer
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
//默认情况下，p=2。计算2阶范数
val normalizer1 = new Normalizer()
val normalizer2 = new Normalizer(p = Double.PositiveInfinity)
// Each sample in data1 will be normalized using $L^2$ norm.
val data1 = data.map(x => (x.label, normalizer1.transform(x.features)))
// Each sample in data2 will be normalized using $L^\infty$ norm.
val data2 = data.map(x => (x.label, normalizer2.transform(x.features)))
```
&emsp;&emsp;正则化的实现很简单，我们看它的`transform`方法。

```scala
 override def transform(vector: Vector): Vector = {
    //求范数
    val norm = Vectors.norm(vector, p)
    if (norm != 0.0) {
      //稀疏向量可以重用index
      vector match {
        case DenseVector(vs) =>
          val values = vs.clone()
          val size = values.size
          var i = 0
          while (i < size) {
            values(i) /= norm
            i += 1
          }
          Vectors.dense(values)
        case SparseVector(size, ids, vs) =>
          val values = vs.clone()
          val nnz = values.size
          var i = 0
          while (i < nnz) {
            values(i) /= norm
            i += 1
          }
          Vectors.sparse(size, ids, values)
        case v => throw new IllegalArgumentException("Do not support vector type " + v.getClass)
      }
    } else {
      vector
    }
  }
```
&emsp;&emsp;求范数调用了`Vectors.norm`方法，我们可以看看该方法的实现。

```scala
def norm(vector: Vector, p: Double): Double = {
    val values = vector match {
      case DenseVector(vs) => vs
      case SparseVector(n, ids, vs) => vs
      case v => throw new IllegalArgumentException("Do not support vector type " + v.getClass)
    }
    val size = values.length
    if (p == 1) {
      var sum = 0.0
      var i = 0
      while (i < size) {
        sum += math.abs(values(i))
        i += 1
      }
      sum
    } else if (p == 2) {
      var sum = 0.0
      var i = 0
      while (i < size) {
        sum += values(i) * values(i)
        i += 1
      }
      math.sqrt(sum)
    } else if (p == Double.PositiveInfinity) {
      var max = 0.0
      var i = 0
      while (i < size) {
        val value = math.abs(values(i))
        if (value > max) max = value
        i += 1
      }
      max
    } else {
      var sum = 0.0
      var i = 0
      while (i < size) {
        sum += math.pow(math.abs(values(i)), p)
        i += 1
      }
      math.pow(sum, 1.0 / p)
    }
  }
```
&emsp;&emsp;这里分四种情况。当`p=1`时，即计算一阶范数，它的值为所有元素绝对值之和。当`p=2`时，它的值为所有元素的平方和。当`p == Double.PositiveInfinity`时，返回所有元素绝对值的最大值。
如果以上三种情况都不满足，那么按照下面的公式计算。

<div  align="center"><img src="imgs/4.1.png" width = "140" height = "70" alt="4.1" align="center" /></div><br>


