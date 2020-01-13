# coding=utf-8
# created by msg on 2019/11/21 2:03 下午

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

# 加载tf 自带的mnist数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 每张图片是一个矩阵，将图片数据按照行列展开，形成一个向量
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

model = load_model('model/mnist.h5')

pred = model.predict([x_test[0].reshape(1, 784)], )

print('real: {}, predict: {}'.format(y_test[0], tf.argmax(pred[0])))
