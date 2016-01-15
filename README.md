# spark机器学习算法研究和源码分析

&emsp;&emsp;本项目对`spark ml`包中各种算法的原理加以介绍并且对算法的代码实现进行详细分析，旨在加深自己对机器学习算法的理解，熟悉这些算法的分布式实现方式。

## 本项目支持的spark版本

- **spark1.6**

## 本项目支持的spark版本

&emsp;&emsp;本系列目录如下：

* 协同过滤
    * [交换最小二乘](推荐/交换最小二乘/ALS.md)
* 分类和回归
    * 线性支持向量机
    * 逻辑回归
    * 朴素贝叶斯
    * 决策树
    * 随机森林
* 聚类
    * [k-means||](聚类/k-means/k-means.md)
    * 高斯混合
    * PIC（快速迭代算法）
    * LDA（隐式狄利克雷分布）
    * 流式k-means
* 最优化算法
    * 梯度下降算法
    * L-BFGS
    * [NNLS(非负最小二乘)](最优化算法/非负正则化最小二乘/NNLS.md)
* 特征抽取和转换
    * TF-IDF
    * Word2Vec
    * StandardScaler
    * Normalizer
    * ChiSqSelector
    * ElementwiseProduct
    * PCA(主成分分析)
    
## License

&emsp;&emsp;本文使用的许可见[LICENSE](LICENSE)