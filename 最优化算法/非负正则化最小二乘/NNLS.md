> &emsp;&emsp;`spark`中的非负正则化最小二乘法并不是`wiki`中介绍的[NNLS](https://en.wikipedia.org/wiki/Non-negative_least_squares)的实现，而是做了相应的优化。它使用改进投影梯度法结合共轭梯度法来求解非负最小二乘。
在介绍`spark`的源码之前，我们要先了解什么事最小二乘法以及共轭梯度法。

# 1 最小二乘法

## 1.1 最小二乘问题

&emsp;&emsp;在某些最优化问题中，目标函数由若干个函数的平方和构成，它的一般形式如下所示：

<div  align="center"><img src="imgs/math.1.1.png" width = "335" height = "50" alt="1.1" align="center" /></div><br />

&emsp;&emsp;其中`x=（x1,x2,…,xn）`，一般假设`m>=n`。把极小化这类函数的问题称为最小二乘问题。

<div  align="center"><img src="imgs/math.1.2.png" width = "355" height = "55" alt="1.2" align="center" /></div><br />

&emsp;&emsp;当<img src="http://www.forkosh.com/mathtex.cgi?{f}_{i}(x)">为`x`的线性函数时，称（1.2）为线性最小二乘问题，当<img src="http://www.forkosh.com/mathtex.cgi?{f}_{i}(x)">为`x`的非线性函数时，称（1.2）为非线性最小二乘问题。

## 1.2 线性最小二乘问题

&emsp;&emsp;在公式（1.1）中，假设

<div  align="center"><img src="imgs/math.1.3.png" width = "365" height = "25" alt="1.3" align="center" /></div><br />

&emsp;&emsp;其中，`p`是维列向量，`bi`是实数，这样我们可以用矩阵的形式表示（1.1）式。令

<div  align="center"><img src="imgs/math.1.3.1.png" width = "190" height = "65" alt="1.3" align="center" /></div><br />

&emsp;&emsp;A是`m * n`矩阵，`b`是`m`维列向量。则

<div  align="center"><img src="imgs/math.1.4.png" width = "520" height = "84" alt="1.4" align="center" /></div><br />

&emsp;&emsp;因为`F(x)`是凸的，所以对（1.4）求导可以得到全局极小值，令其导数为0，我们可以得到这个极小值。

<div  align="center"><img src="imgs/math.1.5.png" width = "275" height = "43" alt="1.5" align="center" /></div><br />

&emsp;&emsp;假设`A`为满秩，<img src="http://www.forkosh.com/mathtex.cgi?{A}^{T}{A}">为`n`阶对称正定矩阵，我们可以求得`x`的值为以下的形式：

<div  align="center"><img src="imgs/math.1.6.png" width = "235" height = "25" alt="1.6" align="center" /></div><br />

## 1.3 非线性最小二乘问题

&emsp;&emsp;假设在（1.1）中，<img src="http://www.forkosh.com/mathtex.cgi?{f}_{i}(x)">为非线性函数，且`F(x)`有连续偏导数。由于<img src="http://www.forkosh.com/mathtex.cgi?{f}_{i}(x)">为非线性函数，所以（1.2）中的非线性最小二乘无法套用（1.6）中的公式求得。
解这类问题的基本思想是，通过解一系列线性最小二乘问题求非线性最小二乘问题的解。设<img src="http://www.forkosh.com/mathtex.cgi?{x}^{(k)}">是解的第`k`次近似。在<img src="http://www.forkosh.com/mathtex.cgi?{x}^{(k)}">时，将函数<img src="http://www.forkosh.com/mathtex.cgi?{f}_{i}(x)">线性化，从而将非线性最小二乘转换为线性最小二乘问题，
用（1.6）中的公式求解极小点<img src="http://www.forkosh.com/mathtex.cgi?{x}^{(k+1)}">，把它作为非线性最小二乘问题解的第`k+1`次近似。然后再从<img src="http://www.forkosh.com/mathtex.cgi?{x}^{(k+1)}">出发，继续迭代。下面将来推导迭代公式。令

<div  align="center"><img src="imgs/math.1.7.png" width = "490" height = "68" alt="1.7" align="center" /></div><br />

&emsp;&emsp;上式右端是<img src="http://www.forkosh.com/mathtex.cgi?{f}_{i}(x)">在<img src="http://www.forkosh.com/mathtex.cgi?{x}^{(k)}">处展开的一阶泰勒级数多项式。令

<div  align="center"><img src="imgs/math.1.8.png" width = "290" height = "55" alt="1.8" align="center" /></div><br />

&emsp;&emsp;用`∅(x)`近似`F(x)`，从而用`∅(x)`的极小点作为目标函数`F(x)`的极小点的估计。现在求解线性最小二乘问题

<div  align="center"><img src="imgs/math.1.9.png" width = "220" height = "24" alt="1.9" align="center" /></div><br />

&emsp;&emsp;把（1.9）写成

<div  align="center"><img src="imgs/math.1.10.png" width = "285" height = "24" alt="1.10" align="center" /></div><br />

&emsp;&emsp;在公式（1.10）中，

<div  align="center"><img src="imgs/math.1.10.append1.png" width = "380" height = "190" alt="1.10" align="center" /></div><br />

&emsp;&emsp;将<img src="http://www.forkosh.com/mathtex.cgi?{A}_{k}">和`b`带入公式（1.5）中，可以得到，

<div  align="center"><img src="imgs/math.1.11.png" width = "335" height = "25" alt="1.11" align="center" /></div><br />

&emsp;&emsp;如果<img src="http://www.forkosh.com/mathtex.cgi?{A}_{k}">为列满秩，且<img src="http://www.forkosh.com/mathtex.cgi?{{A}_{k}}^{T}{A}_{k}">是对称正定矩阵，那么由（1.11）可以得到`x`的极小值。

<div  align="center"><img src="imgs/math.1.12.png" width = "345" height = "25" alt="1.12" align="center" /></div><br />

&emsp;&emsp;可以推导出<img src="http://www.forkosh.com/mathtex.cgi?2{{A}_{k}}^{T}{f}^{(k)}">是目标函数`F(x)`在<img src="http://www.forkosh.com/mathtex.cgi?{x}^{(k)}">处的梯度，<img src="http://www.forkosh.com/mathtex.cgi?2{{A}_{k}}^{T}{A}_{k}">是函数`∅(x)`的海森矩阵。所以（1.12）又可以写为如下形式。

<div  align="center"><img src="imgs/math.1.13.png" width = "325" height = "29" alt="1.13" align="center" /></div><br />

&emsp;&emsp;公式（1.13）称为`Gauss-Newton`公式。向量
<br />
<div  align="center"><img src="imgs/math.1.14.png" width = "320" height = "25" alt="1.14" align="center" /></div><br />

&emsp;&emsp;称为点<img src="http://www.forkosh.com/mathtex.cgi?{x}^{(k)}">处的`Gauss-Newton`方向。为保证每次迭代能使目标函数值下降（至少不能上升），在求出<img src="http://www.forkosh.com/mathtex.cgi?{d}^{(k)}">后，不直接使用<img src="http://www.forkosh.com/mathtex.cgi?{x}^{(k)}+{d}^{(k)}">作为k+1次近似，而是从<img src="http://www.forkosh.com/mathtex.cgi?{x}^{(k)}">出发，沿<img src="http://www.forkosh.com/mathtex.cgi?{d}^{(k)}">方向进行一维搜索。

<div  align="center"><img src="imgs/math.1.15.png" width = "300" height = "30" alt="1.15" align="center" /></div><br />

&emsp;&emsp;求出步长<img src="http://www.forkosh.com/mathtex.cgi?{\lambda}^{(k)}">后，令

<div  align="center"><img src="imgs/math.1.16.png" width = "300" height = "25" alt="1.16" align="center" /></div><br />

**&emsp;&emsp;最小二乘的计算步骤如下:**

- （1）	给定初始点<img src="http://www.forkosh.com/mathtex.cgi?{x}^{(1)}">，允许误差`ε>0，k=1`

- （2）	计算函数值<img src="http://www.forkosh.com/mathtex.cgi?{f}_{i}(x)">，得到向量<img src="http://www.forkosh.com/mathtex.cgi?{f}^{(k)}">，再计算一阶偏导，得到`m*n`矩阵<img src="http://www.forkosh.com/mathtex.cgi?{A}_{(k)}">

- （3）	解方程组（1.14）求得`Gauss-Newton`方向<img src="http://www.forkosh.com/mathtex.cgi?{d}^{(k)}">

- （4）从<img src="http://www.forkosh.com/mathtex.cgi?{x}^{(k)}">出发，沿着<img src="http://www.forkosh.com/mathtex.cgi?{d}^{(k)}">作一维搜索，求出步长<img src="http://www.forkosh.com/mathtex.cgi?{\lambda}^{(k)}">，并令
<img src="http://www.forkosh.com/mathtex.cgi?{x}^{(k+1)}={x}^{(k)}- \lambda{d}^{(k)}">

- （5）若<img src="http://www.forkosh.com/mathtex.cgi?||{x}^{(k+1)}-{x}^{(k)}|| <= \varepsilon "> 停止迭代，求出`x`，否则，`k=k+1`，返回步骤（2）

&emsp;&emsp;在某些情况下，矩阵<img src="http://www.forkosh.com/mathtex.cgi?{A}^{T}A">是奇异的，这种情况下，我们无法求出它的逆矩阵，因此我们需要对其进行修改。用到的基本技巧是将一个正定对角矩阵添加到<img src="http://www.forkosh.com/mathtex.cgi?{A}^{T}A">上，改变原来矩阵的特征值结构，使其变成条件较好的对称正定矩阵。
典型的算法是`Marquardt`。

<div  align="center"><img src="imgs/math.1.17.png" width = "350" height = "30" alt="1.17" align="center" /></div><br />

&emsp;&emsp;其中，`I`是n阶单位矩阵，`alpha`是一个正实数。当`alpha`为0时，<img src="http://www.forkosh.com/mathtex.cgi?{d}^{(k)}">就是`Gauss-Newton`方向，当`alpha`充分大时，这时<img src="http://www.forkosh.com/mathtex.cgi?{d}^{(k)}">接近`F(x)`在<img src="http://www.forkosh.com/mathtex.cgi?{x}^{(k)}">处的最速下降方向。算法的具体过程见参考文献【1】。

# 2 共轭梯度法

## 2.1 共轭方向

&emsp;&emsp;在讲解共轭梯度法之前，我们需要先知道什么事共轭方向，下面的定义给出了答案。

**定义2.1** 设`A`是`n*n`对称正定矩阵，若两个方向<img src="http://www.forkosh.com/mathtex.cgi?{d}^{(1)}"> 和<img src="http://www.forkosh.com/mathtex.cgi?{d}^{(2)}">满足

<div  align="center"><img src="imgs/math.2.1.png" width = "220" height = "22" alt="2.1" align="center" /></div><br />

则称这两个方向关于`A`共轭。若<img src="http://www.forkosh.com/mathtex.cgi?{d}^{(1)},{d}^{(2)},...,{d}^{(k)}">是`k`个方向，它们两两关于`A`共轭，则称这组方向是关于`A`共轭的。即

<div  align="center"><img src="imgs/math.2.2.png" width = "340" height = "25" alt="2.2" align="center" /></div><br />



