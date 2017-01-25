# 广义线性回归

## 1 普通线性模型

&emsp;&emsp;普通线性模型(`ordinary linear model`)可以用下式表示：

$$Y = {\beta}_0+{\beta}_1x_1+{\beta}_2x_2+…+{\beta}_{p-1}x_{p-1}+\epsilon$$

&emsp;&emsp;这里$\beta$是未知参数，$\epsilon$是误差项。普通线性模型主要有以下几点假设：

- 因变量$Y$和误差项$\epsilon$均服从正太分布。其中$\epsilon \sim N(0,{{\sigma }^{2}})$，$Y\sim N({{\theta }^{T}}x,{{\sigma }^{2}})$。