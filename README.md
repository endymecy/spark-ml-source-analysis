# spark机器学习算法研究和源码分析

&emsp;&emsp;本项目对`spark ml`包中各种算法的原理加以介绍并且对算法的代码实现进行详细分析，旨在加深自己对机器学习算法的理解，熟悉这些算法的分布式实现方式。

## 本系列文章支持的spark版本

- **spark1.6.x**

## 本系列的目录结构

&emsp;&emsp;本系列目录如下：
* [数据类型](数据类型/data-type.md)
* [基本统计](基本统计/summary-statistics.md)
    * [summary statistics（概括统计）](基本统计/summary-statistics.md)
    * [correlations（相关性系数）](基本统计/correlations.md)
    * [tratified sampling（分层取样）](基本统计/tratified-sampling.md)
    * [hypothesis testing（假设检验）](基本统计/hypothesis-testing.md)
    * [random data generation（随机数生成）](基本统计/random-data-generation.md)
    * [Kernel density estimation（核密度估计）](基本统计/kernel-density-estimation.md)
* [协同过滤](推荐/交换最小二乘/ALS.md)
    * [交换最小二乘](推荐/交换最小二乘/ALS.md)
* [分类和回归](分类和回归/readme.md)
    * [线性模型](分类和回归/线性模型/readme.md)
        * SVMs(支持向量机)
        * [逻辑回归](分类和回归/线性模型/逻辑回归/logic-regression.md)
        * 线性回归
    * 朴素贝叶斯
    * 决策树
    * 多种树
        * 随机森林
        * 梯度增强树
    * 保序回归
* [聚类](聚类/readme.md)
    * [k-means算法](聚类/k-means/k-means.md)
    * [GMM（高斯混合模型）](聚类/gaussian-mixture/gaussian-mixture.md)
    * [PIC（快速迭代聚类）](聚类/PIC/pic.md)
    * [LDA（隐式狄利克雷分布)](聚类/LDA/lda.md)
    * [二分k-means算法](聚类/bis-k-means/bisecting-k-means.md)
    * [流式k-means算法](聚类/streaming-k-means/streaming-k-means.md)
* [最优化算法](最优化算法/梯度下降/gradient-descent.md)
    * [梯度下降算法](最优化算法/梯度下降/gradient-descent.md)
    * [L-BFGS（限制内存BFGS）](最优化算法/L-BFGS/lbfgs.md)
    * [NNLS(非负最小二乘)](最优化算法/非负最小二乘/NNLS.md)
* [降维](降维/SVD/svd.md)
    * [SVD（奇异值分解）](降维/SVD/svd.md)
    * [PCA（主成分分析）](降维/PCA/pca.md)
* [特征抽取和转换](特征抽取和转换/TF-IDF.md)
    * [TF-IDF](特征抽取和转换/TF-IDF.md)
    * [Word2Vec](特征抽取和转换/Word2Vector.md)
    * [StandardScaler（特征缩放）](特征抽取和转换/StandardScaler.md)
    * [Normalizer(规则化)](特征抽取和转换/normalizer.md)
    * [ChiSqSelector(卡方选择器)](特征抽取和转换/chi-square-selector.md)
    * [ElementwiseProduct(元素智能乘积)](特征抽取和转换/element-wise-product.md)
    
## 说明

&emsp;&emsp;本专题的大部分内容来自[spark源码](https://github.com/apache/spark)、[spark官方文档](https://spark.apache.org/docs/latest)，并不用于商业用途。转载请注明本专题地址。
本专题引用他人的内容均列出了参考文献，如有侵权，请务必邮件通知作者。邮箱地址：`endymecy@sina.cn`
    
## License

&emsp;&emsp;本文使用的许可见 [LICENSE](LICENSE)