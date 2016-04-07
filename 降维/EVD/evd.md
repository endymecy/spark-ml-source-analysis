# 特征值分解

&emsp;&emsp;假设向量`v`是方阵`A`的特征向量，可以表示成下面的形式：

<div  align="center"><img src="imgs/1.1.png" width = "85" height = "30" alt="1.1" align="center" /></div><br>

&emsp;&emsp;这里`lambda`表示特征向量`v`所对应的特征值。并且一个矩阵的一组特征向量是一组正交向量。特征值分解是将一个向量分解为下面的形式：

<div  align="center"><img src="imgs/1.2.png" width = "160" height = "25" alt="1.2" align="center" /></div><br>

&emsp;&emsp;其中`Q`是这个矩阵`A`的特征向量组成的矩阵。`sigma`是一个对角矩阵，每个对角线上的元素就是一个特征值。

&emsp;&emsp;特征值分解是一个提取矩阵特征很不错的方法，但是它只适合于方阵，对于非方阵，它不适合。这就需要用到奇异值分解。

