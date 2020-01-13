# coding=utf-8
# created by msg on 2019/11/21 7:33 下午

import tensorflow as tf
from tensorflow.keras.layers import Dense

X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.constant([[10.0], [20.0]])


class Linear(tf.keras.Model):
    """
    自定义一个线性回归模型， 与pytorch类似
    """

    # 初始化代码（包含 call 方法中会用到的层）
    def __init__(self):
        super().__init__()
        self.dense = Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, inputs, training=None, mask=None):
        # 添加模型调用的代码
        output = self.dense(inputs)
        return output


# 构造模型
model = Linear()

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
for i in range(100):
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.reduce_mean(tf.square(y_pred - y))
    grads = tape.gradient(loss, model.variables)
    # model.variables直接取得模型所有的权重参数
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

print([x.numpy() for x in model.variables])
