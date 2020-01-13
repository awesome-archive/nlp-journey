# coding=utf-8
# created by msg on 2019/12/4 2:43 下午

import tensorflow as tf


def dot_product(x, kernel):
    return tf.squeeze(tf.matmul(x, tf.expand_dims(kernel, axis=-1)), axis=-1)


class Attention(tf.keras.layers.Layer):
    def __init__(self, bias=True, **kwargs):
        """
        自定义attention层
        """
        self.supports_masking = True
        self.bias = bias
        super().__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # 添加参数
        self.w = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer='glorot_uniform',
                                 name='{}_W'.format(self.name))
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name))
        self.u = self.add_weight((input_shape[-1],),
                                 initializer='glorot_uniform',
                                 name='{}_u'.format(self.name))
        super().build(input_shape)

    def call(self, x, mask=None):
        uit = dot_product(x, self.w)
        if self.bias:
            uit += self.b
        uit = tf.tanh(uit)

        ait = dot_product(uit, self.u)
        a = tf.exp(ait)

        a /= tf.cast(tf.reduce_sum(a, axis=1, keepdims=True) + tf.keras.backend.epsilon(), tf.float32)

        a = tf.expand_dims(a, axis=-1)
        weighted_input = x * a
        return tf.reduce_sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]
