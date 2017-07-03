# 分类与回归

&emsp;&emsp;`spark.mllib`提供了多种方法用于用于[二分类](http://en.wikipedia.org/wiki/Binary_classification)、[多分类](http://en.wikipedia.org/wiki/Multiclass_classification)以及[回归分析](http://en.wikipedia.org/wiki/Regression_analysis)。
下表介绍了每种问题类型支持的算法。

| 问题类型       | 支持的方法   |
| ------------- |:-------------:|
| 二分类        | 线性SVMs、逻辑回归、决策树、随机森林、梯度增强树、朴素贝叶斯 |
| 多分类        | 逻辑回归、决策树、随机森林、朴素贝叶斯 |
| 回归          | 线性最小二乘、决策树、随机森林、梯度增强树、保序回归 |

&emsp;&emsp;点击链接，了解具体的算法实现。

* 分类和回归
    * [线性模型](线性模型/readme.md)
        * [SVMs(支持向量机)](线性模型/支持向量机/lsvm.md)
        * [逻辑回归](线性模型/逻辑回归/logic-regression.md)
        * [线性回归](线性模型/回归/regression.md)
    * [朴素贝叶斯](朴素贝叶斯/nb.md)
    * [决策树](决策树/decision-tree.md)
    * [组合树](组合树/readme.md)
        * [随机森林](组合树/随机森林/random-forests.md)
        * [梯度提升树](组合树/梯度提升树/gbts.md)
    * [生存回归](生存回归/survival-regression.md)
    * [保序回归](保序回归/isotonic-regression.md)