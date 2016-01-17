# k-means、k-means++以及k-means||

&emsp;&emsp;本文会介绍一般的`k-means`算法、`k-means++`算法以及基于`k-means++`算法的`k-means||`算法。在`spark ml`，已经实现了`k-means`算法以及`k-means||`算法。
本文首先会介绍这三个算法的原理，然后在了解原理的基础上分析`spark`中的实现代码。

## k-means算法原理分析

&emsp;&emsp;`k-means`算法是聚类分析中使用最广泛的算法之一。它把`n`个对象根据它们的属性分为`k`个聚类以便使得所获得的聚类满足：同一聚类中的对象相似度较高；而不同聚类中的对象相似度较小。

&emsp;&emsp;`k-means`算法的基本过程如下所示：

- （1）任意选择`k`个初始中心<img src="http://www.forkosh.com/mathtex.cgi?C={{c}_{1},{c}_{2},...,{c}_{k}}">。
- （2）计算`X`中的每个对象与这些中心对象的距离；并根据最小距离重新对相应对象进行划分；
- （3）重新计算每个中心对象<img src="http://www.forkosh.com/mathtex.cgi?{C}_{i}">的值

    <div  align="center"><img src="imgs/math.1.1.png" width = "200" height = "50" alt="1.1" align="center" /></div><br />

- （4）计算标准测度函数，当满足一定条件，如函数收敛时，则算法终止；如果条件不满足则重复步骤（2），（3）。


