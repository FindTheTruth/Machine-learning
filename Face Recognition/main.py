# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import zipfile as zf
import os
import numpy as np
import struct
from BMPReader import *
from matplotlib import image as mpig
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split


class Utils:
    @staticmethod
    def unzip(path, unzippath='Data'):
        """
        :param unzippath:
        :param path: 图片的zip路径
        :return: list文件
        """
        """读取当前路径下的所有压缩文件"""
        zipfiles = []
        for root, dir, files in os.walk(path):
            for file in files:
                zipfile = os.path.join(root, file)
                if zipfile.endswith('.zip'):
                    zipfiles.append(zipfile)
        print(zipfiles)
        for zipfile in zipfiles:
            with zf.ZipFile(zipfile, 'r') as zipobj:

                zip_list = zipobj.namelist()
                for i in zip_list:
                    if not i.endswith(".bmp"):
                        if os.path.exists(i[:-1]):
                            print("file exists %s,no need to unzip" % (i[:-1]))
                        else:
                            os.mkdir(unzippath+"/"+i[:-1])
                zipobj.extractall(unzippath + "/")

    @staticmethod
    def read_bmp(picture):
        """

        :param picture: path of picture
        :return: rgb list
        """
        picture_pixes = LoadBmp(picture)
        width = len(picture_pixes.pixels)
        height = len(picture_pixes.pixels[0])
        r = [0] * (width * height)
        g = [0] * (width * height)
        b = [0] * (width * height)

        for i in range(width):
            for j in range(height):
                pixes = picture_pixes.pixels[i][j]
                r[i * width + j] = pixes.r
                g[i * width + j] = pixes.g
                b[i * width + j] = pixes.b

        return r, g, b

    def read_bmps(self,path, islabel=False):
        """

        :param islabel: 读取返回的数据是否带有标签，存在则返回标签数据，否则不返回
        :return:
        """
        r_list = []
        g_list = []
        b_list = []
        final_list = []
        label_list = []
        size = 0
        for root, dir, files in os.walk(path):
            for file in files:
                if file.endswith("bmp"):
                    bmp_path= os.path.join(root, file)
                    r,g,b = self.read_bmp(bmp_path)

                    r_list.append(r)
                    g_list.append(g)
                    b_list.append(b)
                    size = size + 1
                    print(size)

                    if islabel:
                        print("label:",int(root[-3:]))
                        label_list.append(int(root[-3:]))

                    if size == 1000:
                        break
            if size == 1000:
                break
        r_list = np.array(r_list).transpose().tolist()
        g_list = np.array(g_list).transpose().tolist()
        b_list = np.array(b_list).transpose().tolist()

        final_list.append(r_list)
        final_list.append(g_list)
        final_list.append(b_list)
        return np.array(final_list).transpose(), np.array(label_list)

    @staticmethod
    def save_png(color,path,shape):
        """
        :param color: rgb三元素存储内容
        :param path:对应图片存储路径和名字
        :param shape:存储图片应该有的像素大小
        :return:无返回
        """
        color = color/255.
        clip =  list(np.clip(np.array(color), 0, 1))
        compressPicture = np.resize(clip, shape)
        mpig.imsave(path, arr=compressPicture)

    @staticmethod
    def save_np(path,data):
        np.save(path, data)

    @staticmethod
    def load_np(path):
        return np.load(path)

class Model:
    def __init__(self, method, **params):
        """

        :param method: "PCA" or “LDA”
        """
        self.method = method
        self.components_ = None
        self.mean_ = None
        if self.method == "PCA":
            print(params)
            self.model = PCA(n_components=params['params']['n_components'])
        if self.method == "LDA":
            self.model = LDA()

    def training(self, x,y=None):
        if self.method == "PCA":
            low_dim_data = self.model.fit_transform(x)
            self.components_ = self.model.components_
            self.mean = self.model.mean_
            return low_dim_data, self.components_, self.mean_
        elif self.method == "LDA":
             low_dim_data = self.model.fit_transform(x,y)
             return low_dim_data

    def predict(self, x):
        if self.method == "PCA":
            new_feature = self.model.transform(x)
        if self.method == "LDA":
            new_feature = self.model.transform(x)
        return new_feature

    def extracted_feature_display(self, picture, path = "./Data/c.png"):

        partpicture = model.predict(np.expand_dims(picture, 0))
        print(np.shape(partpicture),np.shape(self.components_))
        if self.method == "PCA":
            X_old = np.dot(partpicture, self.components_) + self.mean
            X_old = np.array([int(i) for i in X_old[0]])

        Utils.save_png(X_old, path, (640, 480, 3))


if __name__ == '__main__':
    if os.path.exists("Data/picture.npy") and os.path.exists("Data/label.npy"):
        whole_picture = Utils.load_np("Data/picture.npy")
        label = Utils.load_np("Data/label.npy")
        # print(label)
    else:

        whole_picture, label = Utils().read_bmps('./Data', True)
        picture_shape = np.shape(whole_picture)
        whole_picture = np.reshape(whole_picture,(picture_shape[0],picture_shape[1]*picture_shape[2]))
        print(np.shape(whole_picture),np.shape(label))

        Utils().save_np("Data/picture.npy",whole_picture)
        Utils().save_np("Data/label.npy",label)

    X_train, X_test, y_train, y_test = train_test_split(whole_picture, label, train_size=0.8, random_state=3)
    print(np.shape(X_train))
    params = {"n_components": 0.9}
    # ##PCA
    model = Model(method = "PCA", params =params)
    new_X_train, components_, mean_ = model.training(X_train)
    Utils().save_np("/Data/componets.npy",components_)
    Utils().save_np("/Data/mean.py",mean_)
    new_X_test = model.predict(X_test)
    print(np.shape(new_X_train),np.shape(new_X_test), "label:",y_test[0])
    model.extracted_feature_display(X_test[0],"./Data/c.png")
    model.extracted_feature_display(X_test[2],"./Data/d.png")

    # # 逻辑斯蒂回归算法用于预测人脸信息。
    # lr = LogisticRegression(max_iter=500)
    # lr.fit(new_X_train, y_train)
    # predict_label = lr.predict_proba(new_X_test)
    # print(y_test, np.argmax(predict_label,axis=-1))
    # print("acc:", np.sum(np.array(np.argmax(predict_label,axis=-1)) == np.array(y_test)) / len(y_test))

    # LDA
    model = Model(method="LDA", params=params)
    new_X_train = model.training(new_X_train, y_train)

    # # 直接LDA预测
    # new_y_test = model.predict(new_X_test)
    # print(new_y_test,y_test)
    # print("acc:", np.sum(np.array(new_y_test == np.array(y_test)) / len(y_test)))

    # 作为数据处理,输入LR判断
    new_X_test = model.predict(new_X_test)
    print(np.shape(new_X_test), y_test[0])
    # 逻辑斯蒂回归算法用于预测人脸信息。
    lr = LogisticRegression(max_iter=1000)
    lr.fit(new_X_train, y_train)
    predict_label = lr.predict_proba(new_X_test)
    print(y_test, np.argmax(predict_label, axis=-1))
    print("acc:", np.sum(np.array(np.argmax(predict_label, axis=-1)) == np.array(y_test)) / len(y_test))

