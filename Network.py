import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


class Network:
    layers = []
    network_size = 0
    network = None
    model = None

    def build(self):
        print("【新建神经网络】")
        self.buildInputLayer()
        while True:
            print("\t输入要新建的层：")
            print("\t\t1.卷积层")
            print("\t\t2.全连接层")
            print("\t\t3.池化层")
            print("\t\t4.归一化层")
            print("\t\t5.结束构建")
            print("\t\t", end='')
            userinput = int(input())
            if userinput == 1:  # 卷积层
                self.buildConvLayer()
            elif userinput == 2:  # 全连接层
                self.buildFCLayer()
            elif userinput == 3:  # 池化层
                self.buildPoolingLayer()
            elif userinput == 4:  # 归一化层
                self.buildNormalizeLayer()
            elif userinput == 5:  # 输出层
                self.buildRegressionLayer()
                break

    def train(self, inputs, outputs):
        print("【开始训练神经网络】")
        self.createModel(inputs, outputs)

    def buildInputLayer(self):  # 输入层
        self.layers.append("输入层")
        print("\t【构建输入层】")
        print("\t\t请分别输入数据的尺寸（x,y,z），不存在的维度请输入1：", end='')
        userinput = input().split(" ")
        self.network = input_data(shape=[None, int(userinput[0]), int(userinput[1]), int(userinput[2])])
        print("\t构建完成")

    def buildConvLayer(self):  # 卷积层
        self.layers.append("卷积层")
        print("\t【构建卷积层】")
        print("\t\t请输入过滤器（卷积核）的个数：", end='')
        userinput1 = int(input())
        print("\t\t请输入卷积核大小：", end='')
        userinput2 = int(input())
        print("\t\t请输入卷积核的步长：", end='')
        userinput3 = int(input())
        print("\t\t请输入本卷积层使用的激活函数")
        print("\t\t请在linear,tanh,sigmoid,softmax,softplus,softsign,relu,relu6,leaky_relu,prelu,elu,crelu,selu中选择")
        print("\t\t", end='')
        userinput4 = input()
        self.network = conv_2d(self.network, userinput1, userinput2, strides=userinput3, activation=userinput4)
        print("\t构建完成")

    def buildFCLayer(self):  # 全连接层
        self.layers.append("全连接层")
        print("\t【构建全连接层】")
        print("\t\t请输入节点数量：", end='')
        userinput1 = int(input())
        print("\t\t请输入本全连接层使用的激活函数")
        print("\t\t请在linear,tanh,sigmoid,softmax,softplus,softsign,relu,relu6,leaky_relu,prelu,elu,crelu,selu中选择")
        print("\t\t", end='')
        userinput2 = input()
        self.network = fully_connected(self.network, userinput1, activation=userinput2)
        print("\t构建完成")

    def buildPoolingLayer(self):  # 池化层
        self.layers.append("池化层")
        print("\t【构建池化层】")
        print("\t\t请输入池化核大小：", end='')
        userinput2 = int(input())
        print("\t\t请输入池化核的步长：", end='')
        userinput3 = int(input())
        self.network = max_pool_2d(self.network, userinput2, strides=userinput3)
        print("\t构建完成")

    def buildNormalizeLayer(self):  # 归一化层、
        self.layers.append("归一化层")
        print("\t【构建池化层】")
        self.network = local_response_normalization(self.network)
        print("\t构建完成")

    def buildRegressionLayer(self):  # 输出层
        self.layers.append("输出层")
        print("\t【构建拟合层】")
        self.network = regression(self.network, optimizer='momentum', loss='categorical_crossentropy',
                                  learning_rate=0.001)
        print("\t构建完成")

    def createModel(self, inputs, outputs):  # inputs为训练集的输入数据 outputs为训练集的正确输出
        print("\t【构建训练模型】")
        print("\t\t请输入n_epoch（所有样本迭代次数）：", end='')
        userinput1 = int(input())
        print("\t\t请输入batch_size（每次使用迭代样本数）：", end='')
        userinput2 = int(input())
        self.model = tflearn.DNN(self.network, checkpoint_path='model_alexnet',
                                 max_checkpoints=1, tensorboard_verbose=2)
        self.model.fit(inputs, outputs, n_epoch=userinput1, validation_set=0.1, shuffle=True,
                       show_metric=True, batch_size=userinput2, snapshot_step=200,
                       snapshot_epoch=False, run_id='VAI-NN')
        print("构建完成！开始训练神经网络")
