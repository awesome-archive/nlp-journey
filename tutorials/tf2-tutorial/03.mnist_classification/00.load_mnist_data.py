# coding=utf-8
# created by msg on 2019/11/21 1:45 下午

from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# 加载tf 自带的mnist数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 打印图片
plt.imshow(x_train[0])
plt.title(str(y_train[0]))
plt.show()

# 每张图片是一个矩阵，将图片数据按照行列展开，形成一个向量
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255
