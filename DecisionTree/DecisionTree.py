import numpy as np
import operator
from math import log

"""
   1.entropy相关的计算要重新置0.
   2.numpy数组不能删除index，可以通过hstack重新拼接
   3.如果出现重复执行决策树不唯一的情况，可能是由于选取特征点的损失函数相同，而dict访问是随机的
   4.cart算法的剪枝较为特殊，需要单独处理
"""


class DecisionTree:
    def __init__(self, prePruning, postPruning, algorithm, islog=True,RegressorStopLoss = 1e-3):
        """
        :param prePruning: bool ---> true or false
        :param postPruning: bool ---> true or false
        :param algorithm: 'ID3',‘C4.5’,'ClassCART','RegressorCART'
        :param islog : log detail
        :param RegressorStopLoss: for CARTRegressor to stop iteration
        """

        self.prePruning = prePruning
        self.postPruning = postPruning
        self.method = algorithm
        self.islog = islog

        if self.method == 'ID3':
            self.selectMethod = self.cal_best_features_ID3
            self.cal_data_loss = self.cal_data_entropy
        elif self.method == 'C4.5':
            self.selectMethod = self.cal_best_features_C45
            self.cal_data_loss = self.cal_data_entropy
        elif self.method == 'ClassCART':
            self.selectMethod = self.cal_best_features_ClassCART
            self.cal_data_loss = self.cal_data_GINI
        elif self.method == 'RegressorCART':
            self.selectMethod = self.cal_best_features_RegressorCART
            self.cal_data_loss = self.cal_data_MSE
            self.RegressorStopLoss = RegressorStopLoss
        else:
            print("unsupported method")
            raise ValueError

    def cal_data_entropy(self, output):
        """

        :param output: numpy -->(n_samples,):label of dataset;
        :return: EntropyValue
        """
        unique_vale = set(output)
        length = len(output)
        EntropyValue = 0
        for i in unique_vale:
            v = np.sum(output == i) + 1e-4
            EntropyValue += -(v / length) * log(v / length, 2)

        return EntropyValue

    def cal_data_GINI(self, output):
        """

        :param output: numpy -->(n_samples,):label of dataset;
        :return: GiniValue
        """
        unique_value = set(output)
        length = len(output)
        GiniValue = 0
        for i in unique_value:
            v = np.sum(output == i)
            v = float(v / length)
            GiniValue += v * (1 - v)

        return GiniValue

    def cal_data_MSE(self, output):
        """

        :param output:  numpy_str -->(n_samples,):label of dataset
        :return: return MSE
        """
        if len(output) == 0:
            return 0
        output = np.array(output.copy(), int)
        avg = np.sum(output)/len(output)
        mse = np.sum(np.power(output - avg, 2))
        return mse

    def cal_best_features_ID3(self, dataset, output):
        """

        :param dataset: numpy --> [n_samples,feature]
        :param output: numpy -->[n_samples,]
        :return: index of best feature
        """
        entropy = self.cal_data_entropy(output)
        data_len = len(output)
        bestFeatureIndex = 0
        bestInfoGain = 0
        splitEntropy = 0

        for feature_index in range(len(dataset[0])):
            uniqueValue = set(dataset[:, feature_index])
            splitEntropy = 0
            for feature_value in uniqueValue:
                dataset1 = output[dataset[:, feature_index] == feature_value]
                # if self.islog:
                #     print(" split data len", len(dataset1), "whole data", data_len)
                splitEntropy += (float(len(dataset1)) / data_len * self.cal_data_entropy(dataset1))
            if self.islog:
                print("entropy before,%.3f,split entropy after,%.3f" % (entropy, splitEntropy))
            infoGain = entropy - splitEntropy
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeatureIndex = feature_index

            if self.islog:
                print("cal feature %d" % feature_index)
                print("infoGain %.3f" % infoGain)

        return bestFeatureIndex

    def cal_best_features_C45(self, dataset, output):

        """
         :param dataset: numpy --> [n_samples,feature]
        :param output: numpy -->[n_samples,]
        :return: index of best feature
        """
        entropy = self.cal_data_entropy(output)
        data_len = len(output)
        bestFeatureIndex = 0
        bestInfoGainRate = 0
        splitEntropy = 0
        dataEntropy = 0
        for feature_index in range(len(dataset[0])):
            uniqueValue = set(dataset[:, feature_index])
            splitEntropy = 0
            dataEntropy = 0
            for feature_value in uniqueValue:
                dataset1 = output[dataset[:, feature_index] == feature_value]
                splitEntropy += len(dataset1) / data_len * self.cal_data_entropy(dataset1)
                dataEntropy += -len(dataset1) / data_len * log(len(dataset1) / data_len, 2)

            infoGainRate = (entropy - splitEntropy) / dataEntropy
            if infoGainRate > bestInfoGainRate:
                bestInfoGainRate = infoGainRate
                bestFeatureIndex = feature_index

            if self.islog:
                print("cal feature %d,%s" % (feature_index, feature_value))
                print("infoGainRate %.3f" % infoGainRate)
        return bestFeatureIndex

    def cal_best_features_ClassCART(self, dataset, output):
        """
                :param dataset: numpy --> [n_samples,feature]
               :param output: numpy -->[n_samples,]
               :return: index of best feature and best feature
        """

        bestFeatureValue = 0
        """Gini algorithm """
        data_len = len(output)
        bestFeatureIndex = 0
        bestGini = 1.0
        for feature_index in range(len(dataset[0])):
            uniqueValue = set(dataset[:, feature_index])
            # print("select unique",uniqueValue,sorted(uniqueValue))
            for feature_value in sorted(uniqueValue):

                dataset1 = output[dataset[:, feature_index] <= feature_value]
                dataset2 = output[dataset[:, feature_index] > feature_value]
                splitGini = float(len(dataset1)) / data_len * self.cal_data_GINI(dataset1) + \
                            float(len(dataset2)) / data_len * self.cal_data_GINI(dataset2)

                if splitGini < bestGini:
                    bestGini = splitGini
                    bestFeatureIndex = feature_index
                    bestFeatureValue = feature_value

                if self.islog:
                    print("cal feature %d,%s" % (feature_index, feature_value))
                    print("Gini results %.3f" % splitGini)
        return bestFeatureIndex, bestFeatureValue

    def cal_best_features_RegressorCART(self, dataset, output):
        """
                :param dataset: numpy --> [n_samples,feature]
               :param output: numpy -->[n_samples,]
               :return: index of best feature and best feature
        """

        bestFeatureValue = 0
        """Gini algorithm """
        data_len = len(output)
        bestFeatureIndex = 0
        bestMSE = -1.0
        for feature_index in range(len(dataset[0])):
            uniqueValue = set(dataset[:, feature_index])
            for feature_value in sorted(uniqueValue):

                dataset1 = output[dataset[:, feature_index] <= feature_value]
                dataset2 = output[dataset[:, feature_index] > feature_value]
                splitMSE = float(len(dataset1)) / data_len * self.cal_data_MSE(dataset1) + \
                           float(len(dataset2)) / data_len * self.cal_data_MSE(dataset2)

                if bestMSE == -1.0:
                    bestMSE = splitMSE

                elif splitMSE < bestMSE:
                    bestMSE = splitMSE
                    bestFeatureIndex = feature_index
                    bestFeatureValue = feature_value

                if self.islog:
                    print("cal feature %d,%s" % (feature_index, feature_value))
                    print("MSE results %.3f" % splitMSE)
        return bestFeatureIndex, bestFeatureValue

    def main_vote(self, output):
        countDict = {}

        for i in range(len(output)):
            if i not in countDict.keys():
                countDict[output[i]] = 1
            else:
                countDict[output[i]] += 1
        countDict = sorted(countDict.items(), key=operator.itemgetter(1), reverse=True)

        return countDict[0][0]

    def average_results(self,output):
        if len(output) == 0:
            return 0
        output = np.array(output.copy(), int)
        avg = str(np.sum(output)/len(output))
        return avg


    def getsub_discrete_dataset(self, best_index, current_key, dataset, output=None):
        """

        :param best_index: the index of best feature
        :param current_key: the best split value of best feature
        :param dataset: numpy -->[n_samples,features+1] or [n_samples,feature]
        :param output: numpy -->[n_samples,]
        :return: return dataset and output
        """
        if type(output).__name__ == 'NoneType':
            doutput = dataset[:, -1]
        else:
            doutput = output

        new_dataset = (dataset[dataset[:, best_index] == current_key]).copy()
        doutput = doutput[dataset[:, best_index] == current_key]
        new_dataset = np.hstack((new_dataset[:, :best_index], new_dataset[:, best_index + 1:]))

        return new_dataset, doutput

    def getsub_continus_dataset(self, best_index, current_key, dataset, output=None):
        """
        :param best_index: the index of best feature
        :param current_key: the best split value of best feature
        :param dataset: numpy -->[n_samples,features+1] or [n_samples,feature]
        :param output: numpy -->[n_samples,]
        :return: return dataset and output
        """
        if type(output).__name__ == 'NoneType':
            doutput = dataset[:, -1]
        else:
            doutput = output

        new_dataset1 = (dataset[dataset[:, best_index] <= current_key]).copy()
        new_dataset2 = (dataset[dataset[:, best_index] > current_key]).copy()
        doutput1 = doutput[dataset[:, best_index] <= current_key]
        doutput2 = doutput[dataset[:, best_index] > current_key]

        return new_dataset1, new_dataset2, doutput1, doutput2

    def buildTree(self, dataset, output, val_dataset, label_name):
        """

        :param dataset:  numpy --> [n_samples,feature]
        :param output: numpy -->[n_samples,]
        :param val_dataset:  numpy --> [n_samples,feature+1] (last row is label)
        :return: decisionTree
        """
        # only one label, finish build tree
        if self.method == 'ID3' or self.method == 'C4.5' or self.method == 'ClassCART':
            if 1 == len(set(output)) or len(output) == 0:
                return output[0]
            if len(label_name) == 0:
                return self.main_vote(output)
        else:
            if self.cal_data_MSE(output) < self.RegressorStopLoss:
                return self.average_results(output)
        if self.islog:
            print("load dataset,label", np.shape(dataset), np.shape(output))
        bestIndex = self.selectMethod(dataset, output)

        if self.method == "ClassCART" or self.method == "RegressorCART":
            index = bestIndex[0]
            bestValue = bestIndex[1]
            bestIndex = index
            # index_train = list(map(bool,dataset[:, bestIndex]!=bestValue))
            # index_val = list( map(bool,val_dataset[:, bestIndex]!=bestValue))
            # if np.sum(index_train) != 0:
            #     dataset[:,bestIndex][index_train] = '-'
            # if np.sum(index_val) != 0:
            #     val_dataset[:,bestIndex][index_val ] = '-'

        if self.islog:
            print("current select feature and cut point", label_name[bestIndex])

        tree = {label_name[bestIndex]: {}}
        if self.method == 'ID3' or self.method == 'C4.5':
            currentUniqueKey = sorted(set(dataset[:, bestIndex]))
        else:
            currentUniqueKey = [bestValue]
        print(label_name[bestIndex], currentUniqueKey)
        best_label = label_name[bestIndex]

        # prePruning
        if self.prePruning:
            n_samples = len(dataset[:, 0])
            predict_loss_before = self.cal_data_loss(val_dataset[:, -1])
            predict_loss_after = 0
            for currentKey in currentUniqueKey:
                data_len = np.sum(dataset[:, bestIndex] == currentKey)
                val_label = val_dataset[val_dataset[:, bestIndex] == currentKey][:, -1]
                predict_loss_after += (data_len) / n_samples * self.cal_data_loss(val_label)

            if predict_loss_before < predict_loss_after:
                return self.main_vote(output)

        if self.method == 'ID3' or self.method == 'C4.5':
            # print("......")
            for currentKey in currentUniqueKey:
                new_dataset, new_output = self.getsub_discrete_dataset(bestIndex, currentKey, dataset, output)
                new_val_dataset, val_output = self.getsub_discrete_dataset(bestIndex, currentKey, val_dataset)
                new_labelname = label_name.copy()
                del new_labelname[bestIndex]
                tree[best_label][currentKey] = self.buildTree(new_dataset, new_output, new_val_dataset, new_labelname)
        else:
            currentKey = currentUniqueKey[0]
            new_dataset1, new_dataset2, new_output1, new_output2 = self.getsub_continus_dataset(bestIndex, currentKey,
                                                                                                dataset, output)
            new_val_dataset1, new_val_dataset2, val_output1, val_output2 = self.getsub_continus_dataset(bestIndex,
                                                                                                        currentKey,
                                                                                                        val_dataset)
            tree[best_label][currentKey] = self.buildTree(new_dataset1, new_output1, new_val_dataset1, label_name)
            tree[best_label]['+'] = self.buildTree(new_dataset2, new_output2, new_val_dataset2, label_name)

        # postPruning
        if self.postPruning:
            n_samples = len(dataset[:, 0])
            predict_loss_before = self.cal_data_loss(val_dataset[:, -1])
            predict_loss_after = 0
            for currentKey in currentUniqueKey:
                data_len = np.sum(dataset[:, bestIndex] == currentKey)
                val_label = val_dataset[val_dataset[:, bestIndex] == currentKey][:, -1]
                predict_loss_after += (data_len) / n_samples * self.cal_data_loss(val_label)
            if predict_loss_before < predict_loss_after:
                return self.main_vote(output)

        return tree

    def predict_single_input(self, tree, input, label_name):
        firstStr = list(tree.keys())[0]
        secondDict = tree[firstStr]
        predict = 0
        for i in label_name:
            if i == firstStr:
                break
        for key in secondDict.keys():
            if key == input[i]:
                if type(secondDict[key]).__name__ == 'dict':
                    predict = self.predict_single_input(secondDict[key], input, label_name)
                else:
                    predict = secondDict[key]
        return predict

    def predict_whole_dataset(self, tree, dataset, label_name):
        predict = []
        for i in len(dataset[:, 0]):
            predict.append(self.predict_single_input(tree, dataset[i], label_name))
        return predict
