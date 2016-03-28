# L-BFGS

# 1 牛顿法

&emsp;&emsp;设`f(x)`是二次可微实函数，又设<img src="http://www.forkosh.com/mathtex.cgi?{x}^{(k)}">是`f(x)`一个极小点的估计，我们把`f(x)`在<img src="http://www.forkosh.com/mathtex.cgi?{x}^{(k)}">展开成`Taylor`级数，
并去二阶近似。

<div  align="center"><img src="imgs/1.1.png" width = "615" height = "45" alt="1.1" align="center" /></div><br>

&emsp;&emsp;上式中最后一项的中间部分表示`f(x)`在<img src="http://www.forkosh.com/mathtex.cgi?{x}^{(k)}">处的`Hesse`矩阵。令上式等于0，可以的到下式：

<div  align="center"><img src="imgs/1.2.png" width = "280" height = "40" alt="1.2" align="center" /></div><br>

&emsp;&emsp;设`Hesse`矩阵可逆，由上式可以得到牛顿法的迭代公式如下**(1.1)**

<div  align="center"><img src="imgs/1.3.png" width = "280" height = "40" alt="1.3" align="center" /></div><br>

&emsp;&emsp;值得注意 ， 当初始点远离极小点时，牛顿法可能不收敛。原因之一是牛顿方向不一定是下降方向，经迭代，目标函数可能上升。此外，即使目标函数下降，得到的点一个不一定沿牛顿方向最好的点或极小点。
因此，我们在牛顿方向上增加一维搜索，提出阻尼牛顿法。其迭代公式是**(1.2)**：

<div  align="center"><img src="imgs/1.4.png" width = "240" height = "60" alt="1.4" align="center" /></div><br>

&emsp;&emsp;其中，`lambda`是由一维搜索（参考文献【1】了解一维搜索）得到的步长，即满足

<div  align="center"><img src="imgs/1.5.png" width = "320" height = "40" alt="1.5" align="center" /></div><br>


# 2 拟牛顿法

## 2.1 拟牛顿条件

&emsp;&emsp;前面介绍了牛顿法，它的突出优点是收敛很快，但是运用牛顿法需要计算二阶偏导数，而且目标函数的`Hesse`矩阵可能非正定。为了克服牛顿法的缺点，人们提出了拟牛顿法，它的基本思想是用不包含二阶导数的矩阵近似牛顿法中的`Hesse`矩阵的逆矩阵。
由于构造近似矩阵的方法不同，因而出现不同的拟牛顿法。

&emsp;&emsp;下面分析怎样构造近似矩阵并用它取代牛顿法中的`Hesse`矩阵的逆。上文**(1.2)**已经给出了牛顿法的迭代公式，为了构造`Hesse`矩阵逆矩阵的近似矩阵<img src="http://www.forkosh.com/mathtex.cgi?{H}_{(k)}">，需要先分析该逆矩阵与一阶导数的关系。




