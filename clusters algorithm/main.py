# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from matplotlib import image as mpig
import scipy
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import kmeans_plusplus
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score


class PictureUtils:
    def __init__(self, read, save, sub=0):
        """
        :param read: read document path
        :param save: write document path
        :param sub: is subsampling
        """
        self.read = read
        self.save = save
        self.img_shape = [0, 0, 0]
        self.picture = None
        self.sub = sub

    def read_picture(self):
        self.picture = mpig.imread(self.read)
        if self.sub == 0:
            self.img_shape = list(np.shape(self.picture))
            print(self.img_shape)
        else:
            self.picture = self.subsample()
            self.img_shape = list(np.shape(self.picture))
            print(self.img_shape)

        stand_picture = np.reshape(self.picture, [-1, self.img_shape[-1]])
        print(np.unique(stand_picture, axis=0),np.size(np.unique(stand_picture, axis=0)))
        return stand_picture

    def subsample(self):
        sub_picture = self.picture[::self.sub, ::self.sub]
        return sub_picture

    def compress_picture(self, cluster, cluster_label):
        print(cluster, cluster_label,np.size(np.unique(cluster, axis=0)))
        clip_cluster = list(np.clip(np.array(cluster), 0, 1))
        compressPicture = []
        for label in cluster_label:
            compressPicture.append(clip_cluster[label])
        compressPicture = np.resize(compressPicture, self.img_shape)
        # print(compressPicture, np.shape(compressPicture))
        mpig.imsave(self.save, arr=compressPicture)


class ModelUtils:
    def __init__(self, modelName, **params):
        params = params['params']
        self.modelName = modelName
        model_dict = {'k_means': KMeans,
                      'GM': GaussianMixture}
        if modelName == 'k_means':
            self.model = model_dict[modelName](init=params['init'], n_clusters=params['n_clusters'])
        elif modelName == 'GM':
            self.model = model_dict[modelName](n_components=params['n_clusters'],covariance_type=params['covariance_type'])
        self.paramsDict = params

        self.data = None
        self.clusterslabel = None

    def get_cluster_center(self):
        if self.modelName == 'k_means':
            return self.model.cluster_centers_
        elif self.modelName == 'GM':
            return self.model.means_

    def fit_predict(self, data):
        self.data = data
        self.clusterslabel = self.model.fit_predict(data)
        return self.get_cluster_center(), self.clusterslabel

    def evaluate_score(self):
        sc = silhouette_score(self.data, self.clusterslabel, sample_size=40)
        print("sc finished")
        ch = calinski_harabasz_score(self.data, self.clusterslabel )
        print("ch finished")
        return sc,ch


def compressFunc(read_path, save_path, **params):

    pictureUtils = PictureUtils(read_path, save_path, 0)
    image = pictureUtils.read_picture()
    params = params['params']
    print(params)
    modelUtils = ModelUtils(modelName=params['modelName'], params=params)
    cluster_centers, clusters_label = modelUtils.fit_predict(image)

    print("training finished")
    sc, ch = modelUtils.evaluate_score()
    print(params['modelName'],":score sc", sc, 'ch', ch)
    pictureUtils.compress_picture(cluster_centers, clusters_label)


if __name__ == '__main__':
    read = 'picture/tiger.png'
    save = 'GenerateDir/newtiger_gm32.png'
    param = {
        'n_clusters': 32,
        'covariance_type': 'spherical',
        'modelName':'GM'
    }
    compressFunc(read_path=read, save_path=save, params=param)

    save = 'GenerateDir/newtiger_k32.png'
    param = {'init': 'k-means++',
              'n_clusters': 32,
              'modelName': 'k_means'
              }
    compressFunc(read_path=read, save_path=save, params = param)

