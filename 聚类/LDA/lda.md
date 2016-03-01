# 隐式狄利克雷分布

&emsp;&emsp;`LDA`是一种概率主题模型：隐含狄利克雷分布（`Latent Dirichlet Allocation`，简称`LDA`）。`LDA`是2003年提出的一种[主题模型](http://zh.wikipedia.org/wiki/%E4%B8%BB%E9%A2%98%E6%A8%A1%E5%9E%8B)，它可以将文档集中每篇文档的主题以概率分布的形式给出。
通过分析一些文档，我们可以抽取出它们的主题（分布），根据主题（分布）进行主题聚类或文本分类。同时，它是一种典型的词袋模型，即一篇文档是由一组词构成，词与词之间没有先后顺序的关系。一篇文档可以包含多个主题，文档中每一个词都由其中的一个主题生成。

&emsp;&emsp;举一个简单的例子，比如假设事先给定了这几个主题：Arts、Budgets、Children、Education，然后通过学习的方式，获取每个主题Topic对应的词语，如下图所示：

<div  align="center"><img src="imgs/topic_words.png" width = "650" height = "300" alt="topic_words" align="center" /></div><br>

&emsp;&emsp;然后以一定的概率选取上述某个主题，再以一定的概率选取那个主题下的某个单词，不断的重复这两步，最终生成如下图所示的一篇文章（不同颜色的词语分别表示不同主题）。

<div  align="center"><img src="imgs/docs.png" width = "600" height = "315" alt="docs" align="center" /></div><br>

&emsp;&emsp;我们看到一篇文章后，往往会推测这篇文章是如何生成的，我们通常认为作者会先确定几个主题，然后围绕这几个主题遣词造句写成全文。`LDA`要干的事情就是根据给定的文档，判断它的主题分别。在`LDA`模型中，生成文档的过程有如下几步：

- 从狄利克雷分布<img src="http://www.forkosh.com/mathtex.cgi?{\alpha}">中生成文档i的主题分布<img src="http://www.forkosh.com/mathtex.cgi?{\theta}_{i}">；

- 从主题的多项式分布<img src="http://www.forkosh.com/mathtex.cgi?{\theta}_{i}">中取样生成文档i第j个词的主题<img src="http://www.forkosh.com/mathtex.cgi?{Z}_{i,j}">；

- 从狄利克雷分布<img src="http://www.forkosh.com/mathtex.cgi?{\eta}">中取样生成主题<img src="http://www.forkosh.com/mathtex.cgi?{Z}_{i,j}">对应的词语分布<img src="http://www.forkosh.com/mathtex.cgi?{\beta}_{i,j}">；

- 从词语的多项式分布<img src="http://www.forkosh.com/mathtex.cgi?{\beta}_{i,j}">中采样最终生成词语<img src="http://www.forkosh.com/mathtex.cgi?{W}_{i,j}">

&emsp;&emsp;`LDA`的图模型结构如下图所示：

<div  align="center"><img src="imgs/LDA.png" width = "415" height = "195" alt="topic_words" align="center" /></div><br>

&emsp;&emsp;`LDA`会涉及很多数学知识，后面的章节我会首先介绍`LDA`涉及的数学知识，然后在这些数学知识的基础上详细讲解`LDA`的原理。

# 1 数学预备

## 1.1 Gamma函数

&emsp;&emsp;在高等数学中，有一个长相奇特的`Gamma`函数

<div  align="center"><img src="imgs/1.1.1.png" width = "240" height = "50" alt="gamma函数" align="center" /></div><br>

&emsp;&emsp;通过分部积分，可以推导`gamma`函数有如下递归性质

<div  align="center"><img src="imgs/1.1.2.png" width = "200" height = "30" alt="gamma函数" align="center" /></div><br>

&emsp;&emsp;通过该递归性质，我们可以很容易证明，`gamma`函数可以被当成阶乘在实数集上的延拓，具有如下性质

<div  align="center"><img src="imgs/1.1.3.png" width = "165" height = "30" alt="gamma函数" align="center" /></div><br>

## 1.2 Digamma函数

&emsp;&emsp;如下函数被称为`Digamma`函数，它是`Gamma`函数对数的一阶导数

<div  align="center"><img src="imgs/1.2.1.png" width = "200" height = "55" alt="digamma函数" align="center" /></div><br>

&emsp;&emsp;这是一个很重要的函数，在涉及`Dirichlet`分布相关的参数的极大似然估计时，往往需要用到这个函数。`Digamma`函数具有如下一个漂亮的性质

<div  align="center"><img src="imgs/1.2.2.png" width = "200" height = "50" alt="digamma函数" align="center" /></div><br>

## 1.3 二项分布（Binomial distribution）

&emsp;&emsp;二项分布是由伯努利分布推出的。伯努利分布，又称两点分布或0-1分布，是一个离散型的随机分布，其中的随机变量只有两类取值，即0或者1。二项分布是重复n次的伯努利试验。简言之，只做一次实验，是伯努利分布，重复做了n次，是二项分布。二项分布的概率密度函数为：

<div  align="center"><img src="imgs/1.3.1.png" width = "260" height = "25" alt="二项分布密度函数" align="center" /></div><br>









