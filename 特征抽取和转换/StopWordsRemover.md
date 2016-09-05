# StopWordsRemover

&emsp;&emsp;[Stop words](https://en.wikipedia.org/wiki/Stop_words)是那些需要从输入数据中排除掉的词。删除这些词的原因是,
这些词出现频繁,并没有携带太多有意义的信息。

&emsp;&emsp;`StopWordsRemover`输入一串句子,将这些输入句子中的停用词全部删掉。停用词列表是通过`stopWords`参数来指定的。
一些语言的默认停用词可以通过调用`StopWordsRemover.loadDefaultStopWords(language)`来获得。可以用的语言选项有`danish`, `dutch`, `english`, `finnish`, `french`, `german`,
`hungarian`, `italian`, `norwegian`, `portuguese`, `russian`, `spanish`, `swedish`以及 `turkish`。参数`caseSensitive`表示是否对大小写敏感,默认为`false`。

## 例子

&emsp;&emsp;假设我们有下面的`DataFrame`,列名为`id`和`raw`。

```
 id | raw
----|----------
 0  | [I, saw, the, red, baloon]
 1  | [Mary, had, a, little, lamb]
```
&emsp;&emsp;把`raw`作为输入列,`filtered`作为输出列,通过应用`StopWordsRemover`我们可以得到下面的结果。

```
 id | raw                         | filtered
----|-----------------------------|--------------------
 0  | [I, saw, the, red, baloon]  |  [saw, red, baloon]
 1  | [Mary, had, a, little, lamb]|[Mary, little, lamb]
```

&emsp;&emsp;下面是代码调用的例子。

```scala
import org.apache.spark.ml.feature.StopWordsRemover

val remover = new StopWordsRemover()
  .setInputCol("raw")
  .setOutputCol("filtered")

val dataSet = spark.createDataFrame(Seq(
  (0, Seq("I", "saw", "the", "red", "baloon")),
  (1, Seq("Mary", "had", "a", "little", "lamb"))
)).toDF("id", "raw")

remover.transform(dataSet).show()
```