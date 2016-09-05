# CountVectorizer

&emsp;&emsp;`CountVectorizer`和`CountVectorizerModel`的目的是帮助我们将文本文档集转换为词频(`token counts`)向量。
当事先没有可用的词典时,`CountVectorizer`可以被当做一个`Estimator`去抽取词汇,并且生成`CountVectorizerModel`。
这个模型通过词汇集为文档生成一个稀疏的表示,这个表示可以作为其它算法的输入,比如`LDA`。
&emsp;&emsp;在训练的过程中,`CountVectorizer`将会选择使用语料中词频个数前`vocabSize`的词。一个可选的参数`minDF`也
会影响训练过程。这个参数表示可以包含在词典中的词的最小个数(如果该参数小于1,则表示比例)。另外一个可选的`boolean`参数控制着输出向量。
如果将它设置为`true`,那么所有的非0词频都会赋值为1。这对离散的概率模型非常有用。

## 举例

&emsp;&emsp;假设我们有下面的`DataFrame`,它的列名分别是`id`和`texts`.

```
id  | texts
----|-------------------------------
 0  | Array("a", "b", "c")
 1  | Array("a", "b", "b", "c", "a")
```

&emsp;&emsp;`texts`列的每一行表示一个类型为`Array[String]`的文档。`CountVectorizer`生成了一个带有词典`(a, b, c)`的`CountVectorizerModel`。
经过转换之后,输出的列为`vector`。

```
 id | texts                           | vector
----|---------------------------------|---------------
 0  | Array("a", "b", "c")            | (3,[0,1,2],[1.0,1.0,1.0])
 1  | Array("a", "b", "b", "c", "a")  | (3,[0,1,2],[2.0,2.0,1.0])
```
&emsp;&emsp;下面是代码调用的方法。

```scala
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}

val df = spark.createDataFrame(Seq(
  (0, Array("a", "b", "c")),
  (1, Array("a", "b", "b", "c", "a"))
)).toDF("id", "words")

// fit a CountVectorizerModel from the corpus
val cvModel: CountVectorizerModel = new CountVectorizer()
  .setInputCol("words")
  .setOutputCol("features")
  .setVocabSize(3)
  .setMinDF(2)
  .fit(df)

// alternatively, define CountVectorizerModel with a-priori vocabulary
val cvm = new CountVectorizerModel(Array("a", "b", "c"))
  .setInputCol("words")
  .setOutputCol("features")

cvModel.transform(df).select("features").show()
```