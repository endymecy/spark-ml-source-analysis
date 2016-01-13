> &emsp;&emsp;`spark`中的非负正则化最小二乘法并不是`wiki`中介绍的[NNLS](https://en.wikipedia.org/wiki/Non-negative_least_squares)的实现，而是做了相应的优化。它使用改进投影梯度法结合共轭梯度法来求解非负最小二乘。
在介绍`spark`的源码之前，我们要先了解什么事最小二乘法以及共轭梯度法。

# 1 最小二乘法

## 1.1 最小二乘问题

&emsp;&emsp;在某些最优化问题中，目标函数由若干个函数的平方和构成，它的一般形式如下所示：

<div  align="center"><img src="imgs/math.1.1.png" width = "335" height = "50" alt="1.1" align="center" /></div>

&emsp;&emsp;其中`x=（x1,x2,…,xn）`，一般假设`m>=n`。把极小化这类函数的问题称为最小二乘问题。

<div  align="center"><img src="imgs/math.1.2.png" width = "355" height = "55" alt="1.2" align="center" /></div>

&emsp;&emsp;当<img src="http://www.forkosh.com/mathtex.cgi?{f}_{i}(x)">为`x`的线性函数时，称（1.2）为线性最小二乘问题，当<img src="http://www.forkosh.com/mathtex.cgi?{f}_{i}(x)">为`x`的非线性函数时，称（1.2）为非线性最小二乘问题。

## 1.2 线性最小二乘问题

&emsp;&emsp;在公式（1.1）中，假设

<div  align="center"><img src="imgs/math.1.3.png" width = "365" height = "25" alt="1.3" align="center" /></div>

&emsp;&emsp;其中，`p`是维列向量，`bi`是实数，这样我们可以用矩阵的形式表示（1.1）式。令

<div  align="center"><img src="imgs/math.1.3.1.png" width = "190" height = "65" alt="1.3" align="center" /></div>

&emsp;&emsp;A是`m * n`矩阵，`b`是`m`维列向量。则

<div  align="center"><img src="imgs/math.1.4.png" width = "520" height = "84" alt="1.4" align="center" /></div>

&emsp;&emsp;因为`F(x)`是凸的，所以对（1.4）求导可以得到全局极小值，令其导数为0，我们可以得到这个极小值。

<div  align="center"><img src="imgs/math.1.5.png" width = "220" height = "43" alt="1.5" align="center" /></div>

假设`A`为满秩，<img src="http://www.forkosh.com/mathtex.cgi?{A}^{T}{A}">为n阶对称正定矩阵，我们可以求得x的值为以下的形式：

<div  align="center"><img src="imgs/math.1.6.png" width = "235" height = "25" alt="1.6" align="center" /></div>

## 1.3 非线性最小二乘问题

