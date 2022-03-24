# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from svm_model import  SVM_Model
from svm_model import  SVM_Model
from sklearn.datasets import load_iris


def svm_test():
    # input = np.array([[1,1],[1,2],[0.5,0.5],[3,3],[1.7,2],[2.5,3.0]])
    # output = np.array([1, -1, 1, 1, -1, -1])
    # print(np.shape(input[0]),input[0])
    iris_dataset = load_iris()
    # print(np.shape(iris_dataset["data"][]))
    input = np.array(iris_dataset['data'][:,:2][:100])
    output = np.array(iris_dataset['target'][:100])
    output[output != 1] = -1
    # print(input[:, :2], output)
    print(np.shape(input),np.shape(output))
    inputmean = np.mean(input,axis=0)
    inputstd = np.std(input,axis=0)
    input = (input - inputmean)/inputstd

    model = SVM_Model(input,output,1e-1,0.1,1e-5,'rbf',100)
    model.training()
    model.plot_classification()

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    svm_test()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
