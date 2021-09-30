# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import operator
from BaseUtils import BaseUtils
from DecisionTree import  DecisionTree
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_set, label_name = BaseUtils.read_dataset("./Dataset/dataset.txt")
    test_set, label_name = BaseUtils.read_dataset("./Dataset/testset.txt")
    train_set = np.array(train_set)
    test_set = np.array(test_set)
    decisionTree = DecisionTree(True, True, 'ID3', 1, True)
    tree = decisionTree.buildTree(train_set[:, :-1], train_set[:, -1], test_set, label_name)
    print("final tree is", tree)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
