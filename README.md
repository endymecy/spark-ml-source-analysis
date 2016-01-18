# spark机器学习算法研究和源码分析

&emsp;&emsp;本项目对`spark ml`包中各种算法的原理加以介绍并且对算法的代码实现进行详细分析，旨在加深自己对机器学习算法的理解，熟悉这些算法的分布式实现方式。

## 本系列文章支持的spark版本

- **spark1.6**

## 本系列的目录结构

&emsp;&emsp;本系列目录如下：

* 协同过滤
    * [交换最小二乘](推荐/交换最小二乘/ALS.md)
* 分类和回归
    * 线性支持向量机
    * 逻辑回归
    * 朴素贝叶斯
    * 决策树
    * 随机森林
    * 保序回归
* 聚类
    * [k-means算法](聚类/k-means/k-means.md)
    * 高斯混合算法
    * PIC（快速迭代算法）
    * LDA（隐式狄利克雷分布）
    * [二分k-means算法](聚类/bis-k-means/bisecting-k-means.md)
    * 流式k-means
* 最优化算法
    * 梯度下降算法
    * L-BFGS
    * [NNLS(非负最小二乘)](最优化算法/非负最小二乘/NNLS.md)
* 降维
    * 奇异值分解（SVD）
    * 主成分分析（PCA）
* 特征抽取和转换
    * TF-IDF
    * Word2Vec
    * StandardScaler
    * Normalizer
    * ChiSqSelector
    * ElementwiseProduct
    
## License

&emsp;&emsp;本文使用的许可见 [LICENSE](LICENSE)