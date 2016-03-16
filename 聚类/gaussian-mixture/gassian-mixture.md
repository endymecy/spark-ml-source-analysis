# 高斯混合模型

>> 现有的高斯模型有单高斯模型（`SGM`）和混合高斯模型（`GMM`）两种。从几何上讲，单高斯分布模型在二维空间上近似于椭圆，在三维空间上近似于椭球。
在很多情况下，属于同一类别的样本点并不满足“椭圆”分布的特性，所以我们需要引入混合高斯模型来解决这种情况。

# 1 单高斯模型

&emsp;&emsp;多维变量`X`服从高斯分布时，它的概率密度函数`PDF`定义如下：

<div  align="center"><img src="imgs/1.1.png" width = "390" height = "50" alt="1.1" align="center" /></div><br />

&emsp;&emsp;在上述定义中,`x`是维数为`d`的样本向量，`mu`是模型期望，`sigma`是模型方差。对于单高斯模型，可以明确训练样本是否属于该高斯模型，所以我们经常将`mu`用训练样本的均值代替，将`sigma`用训练样本的方差代替。
假设训练样本属于类别`C`，那么上面的定义可以修改为下面的形式：

<div  align="center"><img src="imgs/1.2.png" width = "380" height = "50" alt="1.2" align="center" /></div><br />

&emsp;&emsp;这个公式表示样本属于类别`C`的概率。我们可以根据定义的概率阈值来判断样本是否属于某个类别。

# 2 混合高斯模型

&emsp;&emsp;高斯混合模型，顾名思义，就是数据可以看作是从多个高斯分布中生成出来的。从[中心极限定理](https://en.wikipedia.org/wiki/Central_limit_theorem)可以看出，高斯分布这个假设其实是比较合理的。
为什么我们要假设数据是由若干个高斯分布组合而成的，而不假设是其他分布呢？实际上不管是什么分布，只`K`取得足够大，这个`XX Mixture Model`就会变得足够复杂，就可以用来逼近任意连续的概率密度分布。只是因为高斯函数具有良好的计算性能，所`GMM`被广泛地应用。

&emsp;&emsp;每个`GMM`由`K`个高斯分布组成，每个高斯分布称为一个组件（`Component`），这些组件线性加成在一起就组成了`GMM`的概率密度函数：

<div  align="center"><img src="imgs/1.3.png" width = "360" height = "75" alt="1.3" align="center" /></div><br />

&emsp;&emsp;根据上面的式子，如果我们要从`GMM`分布中随机地取一个点，需要两步：

- 随机地在这`K`个组件之中选一个，每个组件被选中的概率实际上就是它的系数`pi`

- 选中了组件之后，再单独地考虑从这个组件的分布中选取一个点。

&emsp;&emsp;怎样用`GMM`来做聚类呢？其实很简单，现在我们有了数据，假定它们是由`GMM`生成出来的，那么我们只要根据数据推出`GMM`的概率分布来就可以了，然后`GMM`的`K`个组件实际上就对应了`K`个聚类了。
在已知概率密度函数的情况下，要估计其中的参数的过程被称作“参数估计”。

&emsp;&emsp;我们可以利用最大似然估计来确定这些参数，`GMM`的似然函数如下：

<div  align="center"><img src="imgs/1.4.png" width = "250" height = "75" alt="1.4" align="center" /></div><br />

&emsp;&emsp;可以用`EM`算法来求解这些参数。`EM`算法求解的过程如下：

- 1 **E-步**。求数据点由各个组件生成的概率（并不是每个组件被选中的概率）。对于每个数据<img src="http://www.forkosh.com/mathtex.cgi?{x}_{i}">来说，它由第`k`个组件生成的概率为：

<div  align="center"><img src="imgs/1.5.png" width = "250" height = "60" alt="1.5" align="center" /></div><br />

&emsp;&emsp;在上面的概率公式中，我们假定`mu`和`sigma`均是已知的，它们的值来自于初始化值或者上一次迭代。

- 2 **M-步**。估计每个组件的参数。由于每个组件都是一个标准的高斯分布，可以很容易分布求出最大似然所对应的参数值：

<div  align="center"><img src="imgs/1.6.png" width = "160" height = "70" alt="1.6" align="center" /></div><br />

<div  align="center"><img src="imgs/1.7.png" width = "100" height = "50" alt="1.7" align="center" /></div><br />

<div  align="center"><img src="imgs/1.8.png" width = "180" height = "70" alt="1.8" align="center" /></div><br />

<div  align="center"><img src="imgs/1.9.png" width = "310" height = "70" alt="1.9" align="center" /></div><br />


