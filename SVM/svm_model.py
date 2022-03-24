import numpy as np
from matplotlib import pyplot as plt


class SVM_Model:
    def __init__(self, input, output, C, gamma, epsilon, kernel, epochs):
        """

        :param input: numpy array--->[samples.features]
        :param output: numpy array --->[samples,] ---> values 0 or 1
        :param C: float---> penalty for soft interval
        :param gamma: float ---> only for rbf kernel
        :param epsilon:float ---> stop epsilon
        :param kernel: string ---> "rbf","linear"
        :param epochs: int ---> max epochs
        """
        self.C = C
        self.sample_num = np.shape(input)[0]
        self.alpha = np.zeros((self.sample_num,))
        # print(self.alpha)
        self.gamma = gamma
        self.epsilon = epsilon
        self.input = input
        self.output = output
        self.b = 0.0
        self.E = np.zeros((self.sample_num, 2))  # cache for faster speed
        self.epochs = epochs

        self.nonZeroAlpha = self.alpha[np.nonzero(self.alpha)[0]]  # 非零的alpha
        self.supportVector = self.input[np.nonzero(self.alpha)[0]]  # 支持向量
        self.supportY = self.output[np.nonzero(self.alpha)]  # 支持向量对应的标签

        if kernel == 'rbf':
            self.kernel = self.rbf_kernel
            self.skernel = self.srbf_kernel
        elif kernel == 'linear':
            self.kernel = self.linear_kernel
            self.skernel = self.slinear_kernel
        else:
            print("Init kernel Failed")

    def rbf_kernel(self, X, xj):
        """

        :param X:
        :param xj:
        :return: narray [samples,1]
        """
        length = np.shape(X)[0]
        res = np.zeros((length, 1))
        for i in range(length):
            res[i] = self.srbf_kernel(X[i], xj)
        return res

    def linear_kernel(self, X, xj):
        """
            :param X:
            :param xj:
            :return: narray [samples,1]
        """
        xj = np.expand_dims(xj, 1)
        res = np.dot(X, xj)
        return res

    def srbf_kernel(self, x1, x2):
        return np.exp(-self.gamma * np.sum(np.square(np.array(x1) - np.array(x2))))

    def slinear_kernel(self, x1, x2):
        # dot is allowed to matrix * scalar, matmul is not allowed
        # print("x1,",np.shape(x1))
        # print("x2",np.shape(x2))
        return np.dot(x1, x2.T)

    def compute_Ek(self, k):
        # print("k",k)
        Ek = np.dot((np.expand_dims(self.alpha, 1) * np.expand_dims(self.output, 1)).T, self.kernel(self.input, self.input[k])) \
             + self.b - self.output[k]
        return Ek

    def update_Ek(self, k):
        self.E[k] = [1, self.compute_Ek(k)]

    def select_variable_j(self, i, Ei):
        self.E[i] = [1, Ei]
        CacheE = np.nonzero(self.E[:, 0])[0]  # validE保存更新状态为1的缓存项的行指标

        if len(CacheE) > 1:
            j = 0
            maxValue = 0
            Ej = 0
            for v in CacheE:  # 寻找最大的|Ei-Ej|
                if v == i:
                    continue

                Ev = self.compute_Ek(v)

                if abs(Ev - Ei) > maxValue:
                    maxValue = abs(Ev - Ei)
                    j = v
                    Ej = Ev
        else:  # 随机选择
            j = i

            while j == i:
                j = int(np.random.uniform(0, self.sample_num))  # 左闭右开，类型转化后0<=j<=N-1

            Ej = self.compute_Ek(j)
        # print(j,Ej)
        return j, Ej

    def select_variable(self, i):
        Ei = self.compute_Ek(i)
        gi = Ei + self.output[i]
        if ((self.output[i] * Ei > self.epsilon and float(self.alpha[i]) > 0) or
                (self.output[i] * Ei < -self.epsilon and float(self.alpha[i]) < self.C)):
            j, Ej = self.select_variable_j(i, Ei)
            oldAlphaI = self.alpha[i]
            oldAlphaJ = self.alpha[j]
            L = 0
            H = self.C
            if self.output[i] != self.output[j]:
                L = max(L, oldAlphaJ - oldAlphaI)
                H = min(H, H + oldAlphaJ - oldAlphaI)

            else:
                L = max(L, oldAlphaI + oldAlphaJ - L)
                H = min(H, oldAlphaJ + oldAlphaI)

            cal_kernel = self.skernel(self.input[i], self.input[i]) + self.skernel(self.input[j], self.input[j]) \
                         - 2 * self.skernel(self.input[i], self.input[j])

            if cal_kernel <= 0:
                print("kernel value error")
                return 0

            updateAlpha = oldAlphaJ + self.output[j] * (Ei - Ej) / (cal_kernel + 1e-3)
            if updateAlpha > H:
                updateAlpha = H
            elif updateAlpha < L:
                updateAlpha = L

            if abs(updateAlpha - oldAlphaJ) < 1e-5:
                return 0

            self.alpha[j] = updateAlpha
            self.update_Ek(j)
            self.alpha[i] = oldAlphaI + self.output[i] * self.output[j] * (oldAlphaJ - updateAlpha)
            # print("select i,j",i ,j,Ei,Ej)
            bj = -Ej - self.output[j] * float(self.skernel(self.input[j], self.input[j])) * (float(self.alpha[j]) - oldAlphaJ) - \
                 self.output[j] * float(self.skernel(self.input[j], self.input[i])) * (float(self.alpha[i]) - oldAlphaI) + self.b

            bi = -Ei - self.output[i] * float(self.skernel(self.input[i], self.input[i])) * (float(self.alpha[i]) - oldAlphaI) - \
                 self.output[i] * float(self.skernel(self.input[j], self.input[i])) * (float(self.alpha[j]) - oldAlphaJ) + self.b

            if 0 < float(self.alpha[i]) < self.C:
                self.b = bi

            elif 0 < float(self.alpha[j]) < self.C:
                self.b = bj

            else:
                self.b = 0.5 * (bi + bj)

            return 1

        else:
            return 0

    def findBoundSet(self):
        bound = []
        for i in range(len(self.alpha)):
            if 0 < self.alpha[i] < self.C:
                bound.append(i)
        return bound

    def training(self):
        epoch = 0

        # 初始alpha为0，遍历全集
        isall = True

        while epoch < self.epochs:
            if isall:
                boundchange = 0
                for i in range(self.sample_num):
                    boundchange += self.select_variable(i)

                if boundchange == 0:
                    break
                else:
                    isall = False
            else:
                boundchange = 0
                bound = self.findBoundSet()
                for i in bound:
                    boundchange += self.select_variable(i)

                # 边界值都满足条件时，遍历全集
                if boundchange == 0:
                    isall = True
            epoch = epoch + 1
            print("epoch:", epoch)
            self.update_support_vector()
            print("train acc,%.3f" % self.eval_set(self.input, self.output))

        print("Training ends")

    def update_support_vector(self):
        self.nonZeroAlpha = self.alpha[np.nonzero(self.alpha)[0]]  # 非零的alpha
        self.supportVector = self.input[np.nonzero(self.alpha)[0]]  # 支持向量
        self.supportY = self.output[np.nonzero(self.alpha)]  # 支持向量对应的标签

    def get_support_vector(self):

        return self.supportVector, self.supportY

    def plot_classification(self):
        plt.xlabel('X1')  # 横坐标

        plt.ylabel('X2')  # 纵坐标

        positive = self.input[self.output == 1]
        negative = self.input[self.output == -1]
        # print(positive,negative)
        plt.scatter(positive[:, 0], positive[:, 1], c='r', marker='o')  # +1样本红色标出

        plt.scatter(negative[:, 0], negative[:, 1], c='g', marker='x')  # -1样本绿色标出

        plt.scatter(self.supportVector[:, 0], self.supportVector[:, 1], s=100, c='y', alpha=0.5, marker='o')  # 标出支持向量

        print("支持向量个数:", len(self.supportY))

        X1 = np.arange(-5.0, 5.0, 0.1)

        X2 = np.arange(-5.0, 5.0, 0.1)

        x1, x2 = np.meshgrid(X1, X2)

        # print(x1,x2)

        g = 0

        for i in range(len(self.nonZeroAlpha)):
            if self.kernel == self.rbf_kernel:
                g = g + self.supportY[i] * self.nonZeroAlpha[i] *np.exp(
                ((x1 - self.supportVector[i][0]) ** 2 + (x2 - self.supportVector[i][1]) ** 2) *(-self.gamma))
            else:
                g = g + self.nonZeroAlpha[i] * self.supportY[i] * (x1 * self.supportVector[i][0] + x2 * self.supportVector[i][1])

        g = g + self.b
        plt.contour(x1, x2, g, 0, colors='b')  # 画出超平面
        # plt.legend()

        plt.title("sigma: %f , C: %f" % (self.gamma, self.C))

        plt.show()

    def predict(self, x):
        """
        :param x: [features,]
        :return:
        """

        g = 0

        for i in range(len(self.nonZeroAlpha)):
            g = g + self.nonZeroAlpha[i] * self.supportY[i] * self.skernel(x, self.supportVector[i])

        g = g + self.b
        return g

    def eval_set(self, X, Y):
        """
        :param X: [samples,features]
        :param Y: [samples,]
        :return:  acc of dataset
        """
        """support vector value"""
        for i in range(len(self.supportVector)):
            print(self.supportVector[i],self.supportY[i], self.predict(self.supportVector[i]))
        data_len = len(X)+0.0
        count = 0.0
        for i in range(len(X)):
            if np.sign(self.predict(X[i]))*Y[i] > 0:
                count = count + 1

        return count/data_len

