# 隐式狄利克雷分布

&emsp;&emsp;`LDA`是一种概率主题模型：隐含狄利克雷分布（`Latent Dirichlet Allocation`，简称`LDA`）。`LDA`是2003年提出的一种[主题模型](http://zh.wikipedia.org/wiki/%E4%B8%BB%E9%A2%98%E6%A8%A1%E5%9E%8B)，它可以将文档集中每篇文档的主题以概率分布的形式给出。
通过分析一些文档，我们可以抽取出它们的主题（分布），根据主题（分布）进行主题聚类或文本分类。同时，它是一种典型的词袋模型，即一篇文档是由一组词构成，词与词之间没有先后顺序的关系。一篇文档可以包含多个主题，文档中每一个词都由其中的一个主题生成。

&emsp;&emsp;举一个简单的例子，比如假设事先给定了这几个主题：Arts、Budgets、Children、Education，然后通过学习的方式，获取每个主题Topic对应的词语，如下图所示：

<div  align="center"><img src="imgs/topic_words.png" width = "700" height = "240" alt="topic_words" align="center" /></div><br />

&emsp;&emsp;然后以一定的概率选取上述某个主题，再以一定的概率选取那个主题下的某个单词，不断的重复这两步，最终生成如下图所示的一篇文章（不同颜色的词语分别表示不同主题）。

<div  align="center"><img src="imgs/docs.png" width = "500" height = "315" alt="docs" align="center" /></div><br />

&emsp;&emsp;我们看到一篇文章后，往往会推测这篇文章是如何生成的，我们通常认为作者会先确定几个主题，然后围绕这几个主题遣词造句写成全文。`LDA`要干的事情就是根据给定的文档，判断它的主题分别。在`LDA`模型中，生成文档的过程有如下几步：

- 从狄利克雷分布<img src="http://www.forkosh.com/mathtex.cgi?{\alpha}">中生成文档i的主题分布<img src="http://www.forkosh.com/mathtex.cgi?{\theta}_{i}">；
