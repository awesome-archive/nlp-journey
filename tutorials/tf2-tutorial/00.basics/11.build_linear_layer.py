# coding=utf-8
# created by msg on 2019/11/23 12:44 下午

import tensorflow as tf

"""
自定义层需要继承 tf.keras.layers.Layer 类，
并重写 __init__ 、 build 和 call 三个方法
"""


class LinearLayer(tf.keras.layers.Layer):

    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_variable(name='w',
                                   shape=[input_shape[-1], self.units],
                                   initializer=tf.zeros_initializer())
        self.b = self.add_variable(name='b',
                                   shape=[self.units],
                                   initializer=tf.zeros_initializer())

    def call(self, inputs, **kwargs):
        y_pred = tf.matmul(inputs, self.w) + self.b
        return y_pred
