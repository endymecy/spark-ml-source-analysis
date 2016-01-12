> &emsp;&emsp;`spark`中的非负正则化最小二乘并不是[wiki](https://en.wikipedia.org/wiki/Non-negative_least_squares)中介绍的标准形式，而是做了相应的优化。它使用改进投影梯度法结合共轭梯度法来求解非负最小二乘。
在介绍spark的源码之前，我们要先了解何为最小二乘法和共轭梯度法。

# 1 最小二乘法

## 1.1 最小二乘问题