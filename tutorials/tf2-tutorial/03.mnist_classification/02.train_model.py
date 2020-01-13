# coding=utf-8
# created by msg on 2019/11/21 1:44 下午

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical

# 加载tf 自带的mnist数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 每张图片是一个矩阵，将图片数据按照行列展开，形成一个向量
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

# 将输入变为tensor
inputs = Input(shape=(784,), name='img')

# 三层全连阶层
h1 = Dense(32, activation=tf.nn.relu)(inputs)  # or 'relu'
h2 = Dense(32, activation=tf.nn.relu)(h1)
outputs = Dense(10, activation=tf.nn.softmax)(h2)  # or 'softmax'

# 构造模型
model = Model(inputs=inputs, outputs=outputs, name='mnist_model')

# 输出构造的模型参数等
model.summary()

# 编译模型，设置loss以及优化器等
model.compile(optimizer=RMSprop(),
              loss=sparse_categorical_crossentropy,  # 直接填api，后面会报错
              metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.2)
test_scores = model.evaluate(x_test, y_test, verbose=0)
print('test loss:', test_scores[0])
print('test acc:', test_scores[1])
