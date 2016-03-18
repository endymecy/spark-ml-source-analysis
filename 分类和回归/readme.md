# 分类与回归

&emsp;&emsp;`spark.mllib`提供了多种方法用于用于[二分类](http://en.wikipedia.org/wiki/Binary_classification)、[多分类](http://en.wikipedia.org/wiki/Multiclass_classification)以及[回归分析](http://en.wikipedia.org/wiki/Regression_analysis)。
下表介绍了每种问题类型支持的算法。

| 问题类型       | 支持的方法   |
| ------------- |:-------------:|
| 二分类        | 线性SVMs、逻辑回归、决策树、随机森林、梯度增强树、朴素贝叶斯 |
| 多分类        | 逻辑回归、决策树、随机森林、朴素贝叶斯 |
| 回归          | 线性最小二乘、决策树、随机森林、梯度增强树、保序回归 |

* 分类和回归
    * [线性模型](线性模型/readme.md)
        * SVMs
        * 逻辑回归
        * 线性回归
    * 朴素贝叶斯
    * 决策树
    * [多种树]()
        * 随机森林
        * 梯度增强树
    * 保序回归