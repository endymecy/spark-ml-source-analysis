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

<div  align="center"><img src="imgs/1.3.png" width = "250" height = "95" alt="1.3" align="center" /></div><br>

&emsp;&emsp;`x_i`表示`y_i`测量得到的值。

## 3 算法流程

&emsp;&emsp;这个算法是标准`PAVA`算法的修改版本，它并不从数据的左端开始，而是在需要时连接相邻的点，以产生近似保序最优的顺序。相比一下，`PAVA`对中间的序列并不特别感兴趣，只关心最后的序列。

&emsp;&emsp;有下面一个引理成立。

<div  align="center"><img src="imgs/1.4.png" width = "1000" height = "60" alt="1.3" align="center" /></div><br>

