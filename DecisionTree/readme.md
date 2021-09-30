代码手动实现了决策树的ID3,C4.5以及分类CART和决策CART树，并且实现了预剪枝和后剪枝。

相关参数介绍如下：
DecisionTree类：功能实现类。
        
        :param prePruning[是否预剪枝]: bool ---> true or false
        
        :param postPruning[是否后剪枝]: bool ---> true or false
        
        :param algorithm[采用何种算法]: 'ID3',‘C4.5’,'ClassCART','RegressorCART'
        
        :param islog[是否打印详细日志] : log detail
        
        :param RegressorStopLoss[CART回归树停止迭代的阈值]: for CARTRegressor to stop iteration
        
        
BaseUtils:基础工具实现类，包含了数据集的读取

main:主函数入口

数据集介绍如下：

给出四个属性特征，年龄，工作，是否有房子以及信贷情况来决定是否给予贷款。


![Image text](https://raw.githubusercontent.com/FindTheTruth/Machine-learning/main/DecisionTree/png/1.png)




对数据集进行维度信息编码如下：

       1.年龄：0表示青年，1表示中年，2表示老年；

       2.有工作：0表示无，1表示有

       3.是否有房子： 0表示无，1表示有

       4.信贷情况：0表示一般，1表示好，2表示非常好

贷款结果：

       5. 贷款结果：0表示不给与贷款，1表示给予贷款。
