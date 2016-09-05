# n-gram

&emsp;&emsp;一个[n-gram](https://en.wikipedia.org/wiki/N-gram)是一个包含`n`个`tokens`(如词)的序列。`NGram`可以将输入特征
转换为`n-grams`。

&emsp;&emsp;`NGram`输入一系列的序列,参数`n`用来决定每个`n-gram`的词个数。输出包含一个`n-grams`序列,每个`n-gram`表示一个划定空间的连续词序列。
如果输入序列包含的词少于`n`,将不会有输出。

```scala
import org.apache.spark.ml.feature.NGram

val wordDataFrame = spark.createDataFrame(Seq(
  (0, Array("Hi", "I", "heard", "about", "Spark")),
  (1, Array("I", "wish", "Java", "could", "use", "case", "classes")),
  (2, Array("Logistic", "regression", "models", "are", "neat"))
)).toDF("label", "words")

val ngram = new NGram().setInputCol("words").setOutputCol("ngrams")
val ngramDataFrame = ngram.transform(wordDataFrame)
ngramDataFrame.take(3).map(_.getAs[Stream[String]]("ngrams").toList).foreach(println)
```