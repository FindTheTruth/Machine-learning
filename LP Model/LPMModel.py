import numpy as np
import matplotlib.pyplot as plt
class LPMModel():
    """
    input:
        x -----(feature_num, batch)
        y -------(1,batch)
    """
    def __init__(self, x, y,epochs,lr,eplison):
        self.x = x
        self.y = y
        print("input:x",np.shape(x),"input:y",np.shape(y))
        self.w = np.random.rand(1, len(self.x))
        # self.w = [[0.70,-1.0]]
        self.b = 0.0
        self.epochs = epochs
        self.lr = lr
        self.eplison = eplison
        self.loss_list = []

    def predict_model(self):
        """
          self.predict ----[1.batch]
          self.predictClas ---[1,batch]
        :return:
        """
        self.predict = np.matmul(self.w, self.x) + self.b
        self.predictClas = self.predict.copy()
        self.predictClas[self.predict>=0] = 1.0
        self.predictClas[self.predict<0] = -1.0

    def cal_loss_and_gradient(self):
        equal = np.equal(np.array(self.predictClas),np.array(self.y))
        self.missClasY = np.expand_dims(self.y[~equal],0)
        self.missClasX = []
        for i in range(len(self.x)):
            k = self.x[i]
            self.missClasX.append(k[(~equal)[0]])
        # print(np.shape(self.missClasY),np.shape(self.missClasX))
        self.loss = -np.sum(np.matmul(self.missClasY,np.transpose(self.predict[self.predictClas != self.y])))
        self.w_gradient = -np.matmul(self.missClasY, np.transpose(self.missClasX))
        self.b_gradient = -np.sum(self.missClasY)

    def update_gradient(self,lr):

        self.w = self.w -lr* self.w_gradient
        self.b = self.b -lr*self.b_gradient
        print(self.w,self.b)

    def training(self):
        epoch = 0
        last_loss = 0
        while (epoch <self.epochs):
            self.predict_model()
            self.cal_loss_and_gradient()
            print("epoch:",epoch,"loss", self.loss,"miss clas:",len(self.missClasY[0]))
            self.loss_list.append(self.loss)
            if (self.loss - last_loss< self.eplison or len(self.missClasY[0]) == 0):
                print("training finish")
                break
            else:
                self.update_gradient(self.lr)
            epoch = epoch + 1


    def plot_loss(self):
        x = np.arange(0,70)
        # print(np.shape(self.loss))
        plt.xlabel("Epochs")
        plt.ylabel("Training loss")
        plt.legend()
        plt.plot(x, self.loss_list, linestyle='--')
        plt.show()

    # def plot_curve(self):
    #     plt.scatter(self.x[self.y == 0], self.x[self.y == 0], marker='x', label='品种1')
    #     plt.scatter(self.x[self.y == 1], self.x[self.y == 1], marker='o', label='品种2')
    #     x = np.arange(4, 8)
    #     plt.plot(x, (x * 0.7 - 0.48), linestyle='--')
    #     plt.show()