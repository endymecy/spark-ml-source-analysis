# 奇异值分解

## 1 特征值分解

&emsp;&emsp;假设向量`v`是方阵`A`的特征向量，可以表示成下面的形式：

<div  align="center"><img src="imgs/1.1.png" width = "85" height = "30" alt="1.1" align="center" /></div><br>

&emsp;&emsp;这里`lambda`表示特征向量`v`所对应的特征值。并且一个矩阵的一组特征向量是一组正交向量。特征值分解是将一个向量分解为下面的形式：

<div  align="center"><img src="imgs/1.2.png" width = "160" height = "25" alt="1.2" align="center" /></div><br>

&emsp;&emsp;其中`Q`是这个矩阵`A`的特征向量组成的矩阵。`sigma`是一个对角矩阵，每个对角线上的元素就是一个特征值。

&emsp;&emsp;特征值分解是一个提取矩阵特征很不错的方法，但是它只适合于方阵，对于非方阵，它不适合。这就需要用到奇异值分解。

## 2 奇异值分解

&emsp;&emsp;我们知道，将矩阵`A`的转置乘以该矩阵可以得到一个方阵。利用上面的公式可以得到：

<div  align="center"><img src="imgs/1.3.png" width = "110" height = "25" alt="1.3" align="center" /></div><br>

&emsp;&emsp;现在假设存在`M*N`矩阵`A`，我们的目标是在`n`维空间中找一组正交基，使得经过`A`变换后还是正交的。假设已经找到这样一组正交基：

<div  align="center"><img src="imgs/1.4.png" width = "120" height = "30" alt="1.4" align="center" /></div><br>

&emsp;&emsp;`A`矩阵可以将这组正则基映射为如下的形式。

<div  align="center"><img src="imgs/1.5.png" width = "150" height = "30" alt="1.5" align="center" /></div><br>

&emsp;&emsp;要使上面的基也为正则基，即使它们两两正交，那么需要满足下面的条件。

<div  align="center"><img src="imgs/1.6.png" width = "150" height = "30" alt="1.6" align="center" /></div><br>

&emsp;&emsp;如果正交基`v`选择为<img src="http://www.forkosh.com/mathtex.cgi?{A}^{T}A">的特征向量的话，由于<img src="http://www.forkosh.com/mathtex.cgi?{A}^{T}A">是对称阵，`v`之间两两正交，那么

<div  align="center"><img src="imgs/1.7.png" width = "345" height = "40" alt="1.7" align="center" /></div><br>

&emsp;&emsp;由于下面的公式成立

<div  align="center"><img src="imgs/1.8.png" width = "300" height = "35" alt="1.8" align="center" /></div><br>

&emsp;&emsp;所以取单位向量

<div  align="center"><img src="imgs/1.9.png" width = "190" height = "60" alt="1.9" align="center" /></div><br>

&emsp;&emsp;可以得到

<div  align="center"><img src="imgs/1.10.png" width = "260" height = "55" alt="1.10" align="center" /></div><br>

&emsp;&emsp;奇异值分解是一个能适用于任意的矩阵的一种分解的方法，它的形式如下：

<div  align="center"><img src="imgs/1.11.png" width = "90" height = "30" alt="1.11" align="center" /></div><br>

&emsp;&emsp;其中，`U`是一个`M*M`的方阵，它包含的向量是正交的，称为左奇异向量。`sigma`是一个`N*M`的对角矩阵，每个对角线上的元素就是一个奇异值。`V`是一个`N*N`的矩阵，它包含的向量是正交的，称为右奇异向量。


## 3 源码分析


## 参考文献

【1】[强大的矩阵奇异值分解(SVD)及其应用](http://www.cnblogs.com/LeftNotEasy/archive/2011/01/19/svd-and-applications.html)

【2】[奇异值分解(SVD)原理详解及推导](http://blog.csdn.net/zhongkejingwang/article/details/43053513)

【3】[A Singularly Valuable Decomposition: The SVD of a Matrix](http://www-users.math.umn.edu/~lerman/math5467/svd.pdf)









