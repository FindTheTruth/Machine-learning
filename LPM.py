#coding:gb2312
from sklearn.datasets import load_iris
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
iris_dataset = load_iris()
data = iris_dataset['data']
label = iris_dataset['target']
print(label)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# print(data, label)

feature_a = data[:,0]
feature_b = data[:,1]

feature_combine = data[:,:2]

import LPMModel

y = label[label!=2]
y[y==0] = -1.0
y = np.expand_dims(y,0)
feature_combine = feature_combine[label!=2]
x = np.transpose(feature_combine)


y = y + 0.0

model = LPMModel.LPMModel(x,y,70,1e-3,1e-3)
model.training()
# model.plot_curve()
