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

&emsp;&emsp;设在第`k`次迭代之后，得到<img src="http://www.forkosh.com/mathtex.cgi?{x}^{(k+1)}">，我们将目标函数`f(x)`在点<img src="http://www.forkosh.com/mathtex.cgi?{x}^{(k+1)}">展开成`Taylor`级数，
并取二阶近似，得到

<div  align="center"><img src="imgs/2.1.png" width = "630" height = "50" alt="2.1" align="center" /></div><br>

&emsp;&emsp;由此可知，在<img src="http://www.forkosh.com/mathtex.cgi?{x}^{(k+1)}">附近有，

<div  align="center"><img src="imgs/2.2.png" width = "420" height = "60" alt="2.2" align="center" /></div><br>

&emsp;&emsp;记

<div  align="center"><img src="imgs/2.3.png" width = "240" height = "60" alt="2.3" align="center" /></div><br>

&emsp;&emsp;则有

<div  align="center"><img src="imgs/2.4.png" width = "200" height = "30" alt="2.4" align="center" /></div><br>

&emsp;&emsp;又设`Hesse`矩阵可逆，那么上式可以写为如下形式。

<div  align="center"><img src="imgs/2.5.png" width = "215" height = "35" alt="2.5" align="center" /></div><br>

&emsp;&emsp;这样，计算出`p`和`q`之后，就可以通过上面的式子估计`Hesse`矩阵的逆矩阵。因此，为了用不包含二阶导数的矩阵<img src="http://www.forkosh.com/mathtex.cgi?{H}_{(k+1)}">取代牛顿法中`Hesse`矩阵的逆矩阵，有理由令<img src="http://www.forkosh.com/mathtex.cgi?{H}_{(k+1)}">满足公式**(2.1)**：

<div  align="center"><img src="imgs/2.6.png" width = "140" height = "35" alt="2.6" align="center" /></div><br>

&emsp;&emsp;公式**(2.1)**称为拟牛顿条件。

## 2.2 秩1校正

&emsp;&emsp;当`Hesse`矩阵的逆矩阵是对称正定矩阵时，满足拟牛顿条件的矩阵<img src="http://www.forkosh.com/mathtex.cgi?{H}_{(k)}">也应该是对称正定矩阵。构造这样近似矩阵的一般策略是，
<img src="http://www.forkosh.com/mathtex.cgi?{H}_{(1)}">取为任意一个`n`阶对称正定矩阵，通常选择`n`阶单位矩阵`I`，然后通过修正<img src="http://www.forkosh.com/mathtex.cgi?{H}_{(k)}">给定<img src="http://www.forkosh.com/mathtex.cgi?{H}_{(k+1)}">。
令，

<div  align="center"><img src="imgs/2.7.png" width = "150" height = "30" alt="2.7" align="center" /></div><br>

&emsp;&emsp;秩1校正公式写为如下公式**(2.2)**形式。

<div  align="center"><img src="imgs/2.8.png" width = "360" height = "70" alt="2.8" align="center" /></div><br>

## 2.3 DFP算法

&emsp;&emsp;著名的`DFP`方法是`Davidon`首先提出，后来又被`Feltcher`和`Powell`改进的算法，又称为变尺度法。在这种方法中，定义校正矩阵为公式**(2.3)**

<div  align="center"><img src="imgs/2.9.png" width = "280" height = "60" alt="2.9" align="center" /></div><br>

&emsp;&emsp;那么得到的满足拟牛顿条件的`DFP`公式如下**(2.4)**

<div  align="center"><img src="imgs/2.10.png" width = "320" height = "70" alt="2.10" align="center" /></div><br>

&emsp;&emsp;查看文献【1】，了解`DFP`算法的计算步骤。

## 2.4 BFGS算法









