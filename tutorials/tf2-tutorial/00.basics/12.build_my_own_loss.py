# coding=utf-8
# created by msg on 2019/11/23 12:49 下午

import tensorflow as tf

"""
自定义损失函数需要继承 tf.keras.losses.Loss 类，
重写 call 方法即可，输入真实值 y_true 和模型预测值 y_pred ，
输出模型预测值和真实值之间通过自定义的损失函数计算出的损失值。
"""


class MeanSquaredError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))
