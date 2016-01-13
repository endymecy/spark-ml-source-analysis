> &emsp;&emsp;`spark`中的非负正则化最小二乘法并不是`wiki`中介绍的[NNLS](https://en.wikipedia.org/wiki/Non-negative_least_squares)的实现，而是做了相应的优化。它使用改进投影梯度法结合共轭梯度法来求解非负最小二乘。
在介绍`spark`的源码之前，我们要先了解什么事最小二乘法以及共轭梯度法。

# 1 最小二乘法

## 1.1 最小二乘问题

&emsp;&emsp;在某些最优化问题中，目标函数由若干个函数的平方和构成，它的一般形式如下所示：

<div  align="center"><img src="imgs/math.1.1.png" width = "335" height = "50" alt="1.1" align="center" /></div>
