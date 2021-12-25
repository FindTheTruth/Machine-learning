# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import matplotlib

def read_file(path):
    df = pd.read_csv(path, delimiter='\t', header=None)
    x = df.iloc[:,:-3]
    y = df.iloc[:, -1]
    # print(x.max())
    # x = (x - x.max())/x.max()
    print(x)
    return x,y

def plot_bar(x,y):
    plt.barh(x, y, color=["red", "blue", "purple", "violet", "Chocolate"])
    plt.legend()
    plt.show()


def plot_analysis(neighbor_labels):
    df = pd.DataFrame()
    iters = range(40,100,3)
    df ['room1'] = neighbor_labels[1]
    df ['room2'] = neighbor_labels[2]
    df ['room3'] = neighbor_labels[3]
    df ['room4'] = neighbor_labels[4]
    df ['overall'] = neighbor_labels['overall']

    df ['iter'] = iters
    sns.lineplot(x="iter", y="room1", data=df,label = 'room1')
    sns.lineplot(x="iter", y="room2", data=df,label = 'room2')
    sns.lineplot(x="iter", y="room3", data=df,label = 'room3')
    sns.lineplot(x="iter", y="room4", data=df,label = 'room4')
    sns.lineplot(x="iter", y="overall", data=df,label = 'overall',linewidth=3)
    plt.legend()
    plt.show()

def plot_pie(sizes,labels, explode,pict=True):
    def truevalue(val):
        val = np.round(val / 100. * np.array(sizes).sum(), 0)
        return val
    if pict:
        patches, l_text, p_text = plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                                       labeldistance=1.1, autopct='%2.0f%%', shadow=False,
                                       startangle=90, pctdistance=0.6)
    else:
        patches, l_text, p_text = plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                                          labeldistance=1.1, autopct=truevalue, shadow=False,
                                          startangle=90, pctdistance=0.6)

    # labeldistance，文本的位置离远点有多远，1.1指1.1倍半径的位置
    # autopct，圆里面的文本格式，%3.1f%%表示小数有三位，整数有一位的浮点数
    # shadow，饼是否有阴影
    # startangle，起始角度，0，表示从0开始逆时针转，为第一块。一般选择从90度开始比较好看
    # pctdistance，百分比的text离圆心的距离
    # patches, l_texts, p_texts，为了得到饼图的返回值，p_texts饼图内部文本的，l_texts饼图外label的文本

    # 改变文本的大小
    # 方法是把每一个text遍历。调用set_size方法设置它的属性
    for t in l_text:
        t.set_size = 30
    for t in p_text:
        t.set_size = 20
    # 设置x，y轴刻度一致，这样饼图才能是圆的
    plt.axis('equal')
    plt.legend(loc='upper left', bbox_to_anchor=(-0.1, 1))
    # loc: 表示legend的位置，包括'upper right','upper left','lower right','lower left'等
    # bbox_to_anchor: 表示legend距离图形之间的距离，当出现图形与legend重叠时，可使用bbox_to_anchor进行调整legend的位置
    # 由两个参数决定，第一个参数为legend距离左边的距离，第二个参数为距离下面的距离
    plt.grid()
    plt.show()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = './Data/Unbalance_wifi_localization.txt'
    x, y = read_file(path)
    # print(len(x),len(y),x,y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
    print(pd.DataFrame(y_train).iloc[:, 0].value_counts(),pd.DataFrame(y_test).iloc[:, 0].value_counts())

    neighbor_labels = {}
    for neighbors in range(40,100,3):
        model = KNeighborsClassifier(n_neighbors=neighbors, p=2, metric='minkowski', weights='distance')
        model.fit(X_train, y_train)

        for i in range(1,5):
            select_label_index =  [np.array(y_test)==i]
            if i not in neighbor_labels.keys():
                neighbor_labels[i] = []
            neighbor_labels[i].append(model.score(X_test[select_label_index[0]],y_test[select_label_index[0]]))
            print("room",i," acc:",model.score(X_test[select_label_index[0]],y_test[select_label_index[0]])," len:",np.sum(select_label_index))



        if 'overall' not in neighbor_labels.keys():
            neighbor_labels['overall'] = []
        neighbor_labels['overall'].append(model.score(X_test,y_test,))

print(neighbor_labels)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/


matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False

# plot_analysis(neighbor_labels)
#绘制bar图
x = ["room1","room2","room3","room4","overall"]
y = [0.933, 0.955, 0.966, 1.0, 0.95]
x.reverse()
y.reverse()
plot_bar(x,y)

#绘制饼状图
labels = [u'Room 1', u'Room 2', u'Room 3', u'Room 4']
sizes = [85, 155, 241, 319]
# colors = ['red', 'yellow', 'blue', 'green']
colors = ["pink", "purple", "violet", "Chocolate"]
explode = (0.05, 0, 0, 0)
plot_pie(sizes, labels, explode,False)