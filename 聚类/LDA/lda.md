# 隐式狄利克雷分布

# 前言

&emsp;&emsp;`LDA`是一种概率主题模型：隐含狄利克雷分布（`Latent Dirichlet Allocation`，简称`LDA`）。`LDA`是2003年提出的一种[主题模型](http://zh.wikipedia.org/wiki/%E4%B8%BB%E9%A2%98%E6%A8%A1%E5%9E%8B)，它可以将文档集中每篇文档的主题以概率分布的形式给出。
通过分析一些文档，我们可以抽取出它们的主题（分布），根据主题（分布）进行主题聚类或文本分类。同时，它是一种典型的词袋模型，即一篇文档是由一组词构成，词与词之间没有先后顺序的关系。一篇文档可以包含多个主题，文档中每一个词都由其中的一个主题生成。

&emsp;&emsp;举一个简单的例子，比如假设事先给定了这几个主题：Arts、Budgets、Children、Education，然后通过学习的方式，获取每个主题Topic对应的词语，如下图所示：

<div  align="center"><img src="imgs/topic_words.png" width = "600" height = "300" alt="topic_words" align="center" /></div><br>

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

<div  align="center"><img src="imgs/1.2.1.png" width = "150" height = "40" alt="digamma函数" align="center" /></div><br>

&emsp;&emsp;这是一个很重要的函数，在涉及`Dirichlet`分布相关的参数的极大似然估计时，往往需要用到这个函数。`Digamma`函数具有如下一个漂亮的性质

<div  align="center"><img src="imgs/1.2.2.png" width = "200" height = "50" alt="digamma函数" align="center" /></div><br>

## 1.3 二项分布（Binomial distribution）

&emsp;&emsp;二项分布是由伯努利分布推出的。伯努利分布，又称两点分布或0-1分布，是一个离散型的随机分布，其中的随机变量只有两类取值，即0或者1。二项分布是重复n次的伯努利试验。简言之，只做一次实验，是伯努利分布，重复做了n次，是二项分布。二项分布的概率密度函数为：

<div  align="center"><img src="imgs/1.3.1.png" width = "260" height = "25" alt="二项分布密度函数" align="center" /></div><br>

&emsp;&emsp;对于k=1,2，...,n，其中C(n,k)是二项式系数（这就是二项分布的名称的由来）

<div  align="center"><img src="imgs/1.3.2.png" width = "180" height = "49" alt="二项分布密度函数" align="center" /></div><br>

## 1.4 多项分布

&emsp;&emsp;多项分布是二项分布扩展到多维的情况。多项分布是指单次试验中的随机变量的取值不再是0-1，而是有多种离散值可能（1,2,3...,k）。比如投掷6个面的骰子实验，N次实验结果服从K=6的多项分布。其中：

<div  align="center"><img src="imgs/1.4.1.png" width = "158" height = "69" alt="多项分布" align="center" /></div><br>

&emsp;&emsp;多项分布的概率密度函数为：

<div  align="center"><img src="imgs/1.4.2.png" width = "410" height = "55" alt="多项分布密度函数" align="center" /></div><br>

## 1.5 Beta分布

### 1.5.1 Beta分布

&emsp;&emsp;首先看下面的问题1（问题1到问题4都取自于文献【1】）。

&emsp;&emsp;**问题1：**

<div  align="center"><img src="imgs/question1.png" width = "430" height = "70" alt="问题1" align="center" /></div><br>

&emsp;&emsp; 为解决这个问题，可以尝试计算<img src="http://www.forkosh.com/mathtex.cgi?{x}_{(k)}">落在区间`[x,x+delta x]`的概率。首先，把`[0,1]`区间分成三段`[0,x)`,`[x,x+delta x]`，`(x+delta x,1]`，然后考虑下简单的情形：即假设n个数中只有1个落在了区间`[x,x+delta x]`内，由于这个区间内的数`X(k)`是第k大的，所以`[0,x)`中应该有k−1个数，`(x+delta x,1]`这个区间中应该有n−k个数。
如下图所示：

<div  align="center"><img src="imgs/1.5.1.png" width = "450" height = "140" alt="多项分布密度函数" align="center" /></div><br>

&emsp;&emsp;上述问题可以转换为下述事件E：

<div  align="center"><img src="imgs/1.5.2.png" width = "250" height = "75" alt="事件E" align="center" /></div><br>

&emsp;&emsp;对于上述事件E，有：

<div  align="center"><img src="imgs/1.5.3.png" width = "255" height = "100" alt="事件E" align="center" /></div><br>

&emsp;&emsp;其中，`o(delta x)`表示`delta x`的高阶无穷小。显然，由于不同的排列组合，即n个数中有一个落在`[x,x+delta x]`区间的有n种取法，余下n−1个数中有k−1个落在`[0,x)`的有`C(n-1,k-1)`种组合。所以和事件E等价的事件一共有`nC(n-1,k-1)`个。

&emsp;&emsp;文献【1】中证明，只要落在`[x,x+delta x]`内的数字超过一个，则对应的事件的概率就是`o(delta x)`。所以<img src="http://www.forkosh.com/mathtex.cgi?{x}_{(k)}">的概率密度函数为：

<div  align="center"><img src="imgs/1.5.4.png" width = "340" height = "120" alt="概率密度函数" align="center" /></div><br>

&emsp;&emsp;利用`Gamma`函数，我们可以将f(x)表示成如下形式：

<div  align="center"><img src="imgs/1.5.5.png" width = "260" height = "40" alt="概率密度函数" align="center" /></div><br>

&emsp;&emsp;在上式中，我们用`alpha=k`，`beta=n-k+1`替换，可以得到`beta`分布的概率密度函数

<div  align="center"><img src="imgs/1.5.6.png" width = "230" height = "45" alt="beta分布概率密度函数" align="center" /></div><br>

### 1.5.2 共轭先验分布

&emsp;&emsp;什么是共轭呢？轭的意思是束缚、控制。共轭从字面上理解，则是共同约束，或互相约束。在贝叶斯概率理论中，如果后验概率P(z|x)和先验概率p(z)满足同样的分布，那么，先验分布和后验分布被叫做共轭分布，同时，先验分布叫做似然函数的共轭先验分布。

### 1.5.3 Beta-Binomial 共轭

&emsp;&emsp;我们在问题1的基础上增加一些观测数据，变成**问题2**：

<div  align="center"><img src="imgs/question2.png" width = "500" height = "95" alt="问题2" align="center" /></div><br>

&emsp;&emsp;第2步的条件可以用另外一句话来表述，即“Yi中有m1个比X(k)小，m2个比X(k)大”，所以X(k)是<img src="http://www.forkosh.com/mathtex.cgi?{X}_{(1)},{X}_{(2)},...,{X}_{(n)};{Y}_{(1)},{Y}_{(2)},...,{Y}_{(m)}">中k+m1大的数。

&emsp;&emsp;根据1.5.1的介绍，我们知道事件p服从`beta`分布,它的概率密度函数为：

<div  align="center"><img src="imgs/1.5.7.png" width = "200" height = "20" alt="问题2" align="center" /></div><br>

&emsp;&emsp;按照贝叶斯推理的逻辑，把以上过程整理如下：

- 1、p是我们要猜测的参数，我们推导出p的分布为f(p)=Beta(p|k,n-k+1),称为p的先验分布

- 2、根据Yi中有m1个比p小，有m2个比p大，Yi相当是做了m次伯努利实验，所以m1服从二项分布B(m,p)

- 3、在给定了来自数据提供(m1,m2)知识后，p的后验分布变为f(p|m1,m2)=Beta(p|k+m1,n-k+1+m2)

&emsp;&emsp;贝叶斯估计的基本过程是：

&emsp;&emsp;**先验分布 + 数据的知识 = 后验分布**

&emsp;&emsp;以上贝叶斯分析过程的简单直观的表示就是：

&emsp;&emsp;**Beta(p|k,n-k+1) + BinomCount(m1,m2) = Beta(p|k+m1,n-k+1+m2)**

&emsp;&emsp;更一般的，对于非负实数alpha和beta，我们有如下关系

&emsp;&emsp;**Beta(p|alpha,beta) + BinomCount(m1,m2) = Beta(p|alpha+m1,beta+m2)**

&emsp;&emsp;针对于这种观测到的数据符合二项分布，参数的先验分布和后验分布都是Beta分布的情况，就是`Beta-Binomial`共轭。换言之，`Beta`分布是二项式分布的共轭先验概率分布。二项分布和Beta分布是共轭分布意味着，如果我们为二项分布的参数p选取的先验分布是`Beta`分布，那么以p为参数的二项分布用贝叶斯估计得到的后验分布仍然服从`Beta`分布。

## 1.6 Dirichlet 分布

### 1.6.1 Dirichlet 分布

&emsp;&emsp;`Dirichlet`分布，是`beta`分布在高维度上的推广。`Dirichlet`分布的的密度函数形式跟`beta`分布的密度函数类似：

<div  align="center"><img src="imgs/1.6.1.png" width = "370" height = "75" alt="Dirichlet" align="center" /></div><br>

&emsp;&emsp;其中

<div  align="center"><img src="imgs/1.6.2.png" width = "260" height = "50" alt="Dirichlet" align="center" /></div><br>

&emsp;&emsp;至此，我们可以看到二项分布和多项分布很相似，`Beta`分布和`Dirichlet`分布很相似。并且`Beta`分布是二项式分布的共轭先验概率分布。那么`Dirichlet`分布呢？`Dirichlet`分布是多项式分布的共轭先验概率分布。下文来论证这点。

### 1.6.2 Dirichlet-Multinomial 共轭

&emsp;&emsp;在1.5.3章问题2的基础上，我们更进一步引入**问题3**：

<div  align="center"><img src="imgs/question3.png" width = "320" height = "75" alt="Dirichlet共轭" align="center" /></div><br>

&emsp;&emsp;类似于问题1的推导，我们可以容易推导联合分布。为了简化计算，我们取`x3`满足`x1+x2+x3=1`,`x1`和`x2`是变量。如下图所示。

<div  align="center"><img src="imgs/1.6.3.png" width = "435" height = "75" alt="Dirichlet共轭" align="center" /></div><br>

&emsp;&emsp;概率计算如下：

<div  align="center"><img src="imgs/1.6.4.png" width = "410" height = "120" alt="Dirichlet共轭" align="center" /></div><br>

&emsp;&emsp;于是我们得到联合分布为：

<div  align="center"><img src="imgs/1.6.5.png" width = "450" height = "85" alt="Dirichlet共轭" align="center" /></div><br>

&emsp;&emsp;观察上述式子的最终结果，可以看出上面这个分布其实就是3维形式的`Dirichlet`分布。令`alpha1=k1,alpha2=k2,alpha3=n-k1-k2+1`，分布密度函数可以写为：

<div  align="center"><img src="imgs/1.6.6.png" width = "340" height = "40" alt="Dirichlet共轭" align="center" /></div><br>

&emsp;&emsp;为了论证`Dirichlet`分布是多项式分布的共轭先验概率分布，在上述问题3的基础上再进一步，提出**问题4**。

<div  align="center"><img src="imgs/question4.png" width = "525" height = "165" alt="问题4" align="center" /></div><br>

&emsp;&emsp;为了方便计算，我们记

<div  align="center"><img src="imgs/1.6.7.png" width = "330" height = "23" alt="问题4" align="center" /></div><br>

&emsp;&emsp;根据问题中的信息，我们可以推理得到`p1,p2`在`X;Y`这`m+n`个数中分别成为了第`k1+m1,k1+k2+m1+m2`大的数。后验分布p应该为

<div  align="center"><img src="imgs/1.6.8.png" width = "640" height = "25" alt="问题4" align="center" /></div><br>

&emsp;&emsp;同样的，按照贝叶斯推理的逻辑，可将上述过程整理如下：

- 1 我们要猜测参数`P=(p1,p2,p3)`，其先验分布为`Dir(p|k)`;

- 2 数据`Yi`落到三个区间`[0,p1)`,`[p1,p2]`,`(p2,1]`的个数分别是`m1,m2,m3`,所以`m=(m1,m2,m3)`服从多项分布`Mult(m|p)`;

- 3 在给定了来自数据提供的知识`m`后，`p`的后验分布变为`Dir(P|k+m)`

&emsp;&emsp;上述贝叶斯分析过程的直观表述为：

&emsp;&emsp;**Dir(p|k) + Multcount(m) = Dir(p|k+m)**

&emsp;&emsp;针对于这种观测到的数据符合多项分布，参数的先验分布和后验分布都是`Dirichlet`分布的情况，就是`Dirichlet-Multinomial`共轭。这意味着，如果我们为多项分布的参数p选取的先验分布是`Dirichlet`分布，那么以p为参数的多项分布用贝叶斯估计得到的后验分布仍然服从`Dirichlet`分布。

## 1.7 Beta和Dirichlet分布的一个性质

&emsp;&emsp;如果`p=Beta(t|alpha,beta)`，那么

<div  align="center"><img src="imgs/1.7.1.png" width = "285" height = "130" alt="性质" align="center" /></div><br>

&emsp;&emsp;上式右边的积分对应到概率分布`Beta(t|alpha+1,beta)`，对于这个分布，我们有

<div  align="center"><img src="imgs/1.7.2.png" width = "245" height = "47" alt="性质" align="center" /></div><br>

&emsp;&emsp;把上式带人`E(p)`的计算式，可以得到：

<div  align="center"><img src="imgs/1.7.3.png" width = "233" height = "126" alt="性质" align="center" /></div><br>

&emsp;&emsp;这说明，对于`Beta`分布的随机变量，其期望可以用上式来估计。`Dirichlet`分布也有类似的结论。对于`p=Dir(t|alpha)`，有

<div  align="center"><img src="imgs/1.7.4.png" width = "290" height = "42" alt="性质" align="center" /></div><br>

&emsp;&emsp;这个结论在后文的推导中会用到。

# 参考文献

【1】[LDA数学八卦](http://www.52nlp.cn/lda-math-%E6%B1%87%E6%80%BB-lda%E6%95%B0%E5%AD%A6%E5%85%AB%E5%8D%A6)

【2】[通俗理解LDA主题模型](http://blog.csdn.net/v_july_v/article/details/41209515)







