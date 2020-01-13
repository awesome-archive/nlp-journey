# coding=utf-8
# created by msg on 2019/11/21 1:34 下午

# 导入所需包
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

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

# 打印模型到图片
plot_model(model, 'mnist_model.png')
plot_model(model, 'model_info.png', show_shapes=True)
