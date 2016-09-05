# Tokenizer

&emsp;&emsp;[Tokenization](http://en.wikipedia.org/wiki/Lexical_analysis#Tokenization)是一个将文本(如一个句子)转换为个体单元(如词)的处理过程。
一个简单的`Tokenizer`类就提供了这个功能。下面的例子展示了如何将句子转换为此序列。

&emsp;&emsp;`RegexTokenizer`基于正则表达式匹配提供了更高级的断词(`tokenization`)。默认情况下,参数`pattern`(默认是`\s+`)作为分隔符,
用来切分输入文本。用户可以设置`gaps`参数为`false`用来表明正则参数`pattern`表示`tokens`而不是`splitting gaps`,这个类可以找到所有匹配的事件并作为结果返回。下面是调用的例子。

```scala
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}

val sentenceDataFrame = spark.createDataFrame(Seq(
  (0, "Hi I heard about Spark"),
  (1, "I wish Java could use case classes"),
  (2, "Logistic,regression,models,are,neat")
)).toDF("label", "sentence")

val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
val regexTokenizer = new RegexTokenizer()
  .setInputCol("sentence")
  .setOutputCol("words")
  .setPattern("\\W") // alternatively .setPattern("\\w+").setGaps(false)

val tokenized = tokenizer.transform(sentenceDataFrame)
tokenized.select("words", "label").take(3).foreach(println)
val regexTokenized = regexTokenizer.transform(sentenceDataFrame)
regexTokenized.select("words", "label").take(3).foreach(println)
```