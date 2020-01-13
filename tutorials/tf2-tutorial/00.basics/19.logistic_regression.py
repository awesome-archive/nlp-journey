# coding=utf-8
# created by msg on 2019/11/27 8:55 下午

import tensorflow as tf


# 简单的逻辑回归二分类
class LogisticRegression(tf.keras.Model):

    def __init__(self, units):
        super().__init__()
        self.dense = tf.keras.layers.Dense(units,
                                           activation=tf.keras.activations.sigmoid,
                                           kernel_initializer='glorot_uniform')

    def call(self, inputs):
        y = self.dense(inputs)
        return y


model = LogisticRegression(1)

inputs = tf.Variable([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
y = tf.Variable([[0.], [1.]])

learning_rate = 1e-3
num_epoch = 1000
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
for i in range(num_epoch):
    with tf.GradientTape() as tape:
        y_pred = model(inputs)

        loss = tf.keras.losses.binary_crossentropy(y_pred=y_pred, y_true=y)
        loss = tf.reduce_mean(loss)
        print('batch: {}, loss: {}'.format(i, loss))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

print(model.variables)
