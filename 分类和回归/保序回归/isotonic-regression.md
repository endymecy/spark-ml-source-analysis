# 保序回归

## 1 介绍

&emsp;&emsp;保序回归解决了下面的问题：给定包含`n`个数据点的序列`y_1,y_2,...,y_n`,怎样通过一个单调的序列`beta_1,beta_2,...,beta_n`来归纳这个问题。形式上，这个问题就是为了找到

<div  align="center"><img src="imgs/1.1.png" width = "700" height = "70" alt="1.1" align="center" /></div><br>

&emsp;&emsp;解决这个问题的一个方法就是`pool adjacent violators algorithm(PAVA)`算法。粗略的讲，`PAVA`算法的过程描述如下。

&emsp;&emsp;我们从左边的`y_1`开始，右移`y_1`直到我们遇到第一个违例(`violation`)`y_i < y_i+1`，然后，我们用他们的平方替换他们，并且将这个平方值放到左边以满足单调性。我们继续这个过程，直到我们最后到达`y_n`。

## 2 近似保序

&emsp;&emsp;给定一个序列`y_1,y_2,...,y_n`，我们寻找一个近似单调估计，考虑下面的问题

<div  align="center"><img src="imgs/1.2.png" width = "700" height = "70" alt="1.2" align="center" /></div><br>

&emsp;&emsp;在上式中，`X_+`表示正数部分，即`X_+ = X.1 (x>0)`。这是一个凸优化问题，处罚项处罚违反单调性（即`beta_i > beta_i+1`）的邻近对。

&emsp;&emsp;在公式（2）中，隐含着一个假设，即使用等距的网格测量数据点。如果情况不是这样，那么可以修改惩罚项为下面的形式

<div  align="center"><img src="imgs/1.3.png" width = "200" height = "75" alt="1.3" align="center" /></div><br>

&emsp;&emsp;`x_i`表示`y_i`测量得到的值。

## 3 近似保序算法流程

&emsp;&emsp;这个算法是标准`PAVA`算法的修改版本，它并不从数据的左端开始，而是在需要时连接相邻的点，以产生近似保序最优的顺序。相比一下，`PAVA`对中间的序列并不特别感兴趣，只关心最后的序列。

&emsp;&emsp;有下面一个引理成立。

<div  align="center"><img src="imgs/1.4.png" width = "900" height = "60" alt="1.4" align="center" /></div><br>

&emsp;&emsp;这个引理证明的事实极大地简化了近似保序解路径（`solution path`）的构造。假设在参数值为`lambda`的情况下，有`K_lambda`个连接块，我们用`A_1,A_2,..,A_K_lambda`表示。这样我们可以重写（2）为如下（3）的形式。

<div  align="center"><img src="imgs/1.5.png" width = "700" height = "85" alt="1.5" align="center" /></div><br>

&emsp;&emsp;上面的公式，对`beta`求偏导，可以得到下面的次梯度公式。通过这个公式即可以求得`beta`。

<div  align="center"><img src="imgs/1.6.png" width = "800" height = "65" alt="1.6" align="center" /></div><br>

&emsp;&emsp;为了符合方便，令`s_0 = s_K_lambda = 0`。并且，

<div  align="center"><img src="imgs/1.7.png" width = "300" height = "40" alt="1.7" align="center" /></div><br>

&emsp;&emsp;现在假设，当`lambda`在一个区间内增长时，组`A_1,A_2,...,A_K_lambda`不会改变。我们可以通过相应的`lambda`区分（4）。

<div  align="center"><img src="imgs/1.8.png" width = "600" height = "65" alt="1.8" align="center" /></div><br>

&emsp;&emsp;这个公式的值本身是一个常量，它意味着上式的`beta`是`lambda`的线性函数。

&emsp;&emsp;随着`lambda`的增长，方程（5）将连续的给出解决方案的斜率直到组`A_1,A_2,...,A_K_lambda`改变。更加引理1，只有两个组合并时，这才会发生。`m_i`表示斜率，那么对于每一个`i=1,...,K_lambda - 1`，`A_i`和`A_i+1`合并之后得到的公式如下

<div  align="center"><img src="imgs/1.9.png" width = "600" height = "75" alt="1.9" align="center" /></div><br>

&emsp;&emsp;因此我们可以一直移动，直到`lambda` “下一个”值的到来

<div  align="center"><img src="imgs/1.10.png" width = "600" height = "50" alt="1.10" align="center" /></div><br>

&emsp;&emsp;并且合并`A_i^star`和`A_i^star+1`,其中

<div  align="center"><img src="imgs/1.11.png" width = "600" height = "55" alt="1.11" align="center" /></div><br>

&emsp;&emsp;注意，可能有超过一对组别到达了这个最小值，在这种情况下，会组合所有满足条件的组别。公式（7）和（8）成立的条件是`t_i,i+1`大于`lambda`，如果没有`t_i,i+1`大于`lambda`，说明没有组别可以合并，算法将会终止。

&emsp;&emsp;**算法的流程如下**：

- 初始时，`lambda=0`，`K_lambda=n`,`A_i={i},i=1,2,...,n`。对于每个i，解是`beta_lambda,i = y_i`

- 重复过程

&emsp;&emsp;（1）通过公式（5）计算每个组的斜率`m_i`

&emsp;&emsp;（2）通过公式（6）计算没对相邻组的碰撞次数`t_i,i+1`

&emsp;&emsp;（3）如果`t_i,i+1 < lambda`，终止

&emsp;&emsp;（4）计算公式（7）中的临界点`lambda^star`,并根据斜率更新解

<div  align="center"><img src="imgs/1.12.png" width = "400" height = "55" alt="1.12" align="center" /></div><br>

&emsp;&emsp;对于每个`i`，更加公式（8）合并合适的组别（所以`K_lambda^star = K_lambda - 1`），并设置`lambda = lambda^star`。








