# 朴素贝叶斯

## 1 介绍

&emsp;&emsp;朴素贝叶斯是一种构建分类器的简单方法。该分类器模型会给问题实例分配用特征值表示的类标签，类标签取自有限集合。它不是训练这种分类器的单一算法，而是一系列基于相同原理的算法：所有朴素贝叶斯分类器都假定样本每个特征与其他特征都不相关。
举个例子，如果一种水果其具有红，圆，直径大概3英寸等特征，该水果可以被判定为是苹果。尽管这些特征相互依赖或者有些特征由其他特征决定，然而朴素贝叶斯分类器认为这些属性在判定该水果是否为苹果的概率分布上独立的。

&emsp;&emsp;对于某些类型的概率模型，在有监督学习的样本集中能获取得非常好的分类效果。在许多实际应用中，朴素贝叶斯模型参数估计使用最大似然估计方法；换言之，在不用贝叶斯概率或者任何贝叶斯模型的情况下，朴素贝叶斯模型也能奏效。

&emsp;&emsp;尽管是带着这些朴素思想和过于简单化的假设，但朴素贝叶斯分类器在很多复杂的现实情形中仍能够取得相当好的效果。尽管如此，有论文证明更新的方法（如提升树和随机森林）的性能超过了贝叶斯分类器。

&emsp;&emsp;朴素贝叶斯分类器的一个优势在于只需要根据少量的训练数据估计出必要的参数（变量的均值和方差）。由于变量独立假设，只需要估计各个变量，而不需要确定整个协方差矩阵。

## 2 朴素贝叶斯概率模型

&emsp;&emsp;理论上，概率模型分类器是一个条件概率模型。

<div  align="center"><img src="imgs/1.1.png" width = "130" height = "20" alt="1.1" align="center" /></div>

&emsp;&emsp;独立的类别变量`C`有若干类别，条件依赖于若干特征变量`F_1,F_2,...,F_n`。但问题在于如果特征数量`n`较大或者每个特征能取大量值时，基于概率模型列出概率表变得不现实。所以我们修改这个模型使之变得可行。 贝叶斯定理有以下式子：

<div  align="center"><img src="imgs/1.2.png" width = "350" height = "50" alt="1.2" align="center" /></div>

&emsp;&emsp;实际中，我们只关心分式中的分子部分，因为分母不依赖于`C`而且特征`F_i`的值是给定的，于是分母可以认为是一个常数。这样分子就等价于联合分布模型。
重复使用链式法则，可将该式写成条件概率的形式，如下所示：

<div  align="center"><img src="imgs/1.3.png" width = "765" height = "137" alt="1.3" align="center" /></div>

&emsp;&emsp;现在“朴素”的条件独立假设开始发挥作用:假设每个特征`F_i`对于其他特征`F_j`是条件独立的。这就意味着

<div  align="center"><img src="imgs/1.4.png" width = "180" height = "21" alt="1.4" align="center" /></div>

&emsp;&emsp;所以联合分布模型可以表达为

<div  align="center"><img src="imgs/1.5.png" width = "450" height = "87" alt="1.5" align="center" /></div>

&emsp;&emsp;这意味着上述假设下，类变量`C`的条件分布可以表达为：

<div  align="center"><img src="imgs/1.6.png" width = "310" height = "48" alt="1.6" align="center" /></div>

&emsp;&emsp;其中`Z`是一个只依赖与`F_1,...,F_n`等的缩放因子，当特征变量的值已知时是一个常数。

### 从概率模型中构造分类器

&emsp;&emsp;讨论至此为止我们导出了独立分布特征模型，也就是朴素贝叶斯概率模型。朴素贝叶斯分类器包括了这种模型和相应的决策规则。一个普通的规则就是选出最有可能的那个：这就是大家熟知的最大后验概率（`MAP`）决策准则。相应的分类器便是如下定义的公式：

<div  align="center"><img src="imgs/1.7.png" width = "500" height = "55" alt="1.7" align="center" /></div>


# 参考文献

【1】[朴素贝叶斯分类器](https://zh.wikipedia.org/wiki/%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF%E5%88%86%E7%B1%BB%E5%99%A8)

【2】[Naive Bayes text classification](http://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html)

【3】[The Bernoulli model](http://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html)
