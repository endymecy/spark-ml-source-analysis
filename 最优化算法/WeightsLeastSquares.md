# 带权最小二乘

&emsp;&emsp;给定n个带权的观察样本$(w_i,a_i,b_i)$:

- $w_i$表示第i个观察样本的权重；
- $a_i$表示第i个观察样本的特征向量；
- $b_i$表示第i个观察样本的标签。

&emsp;&emsp;每个观察样本的特征数是m。我们使用下面的带权最小二乘公式作为目标函数：

$$minimize_{x}\frac{1}{2} \sum_{i=1}^n \frac{w_i(a_i^T x -b_i)^2}{\sum_{k=1}^n w_k} + \frac{1}{2}\frac{\lambda}{\delta}\sum_{j=1}^m(\sigma_{j} x_{j})^2$$

