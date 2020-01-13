# coding=utf-8
# created by msg on 2019/11/21 8:53 下午

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *


# mnist数据集加载器
class MnistLoader:

    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        # 处理图片数据常用方式，灰度图片在最后加一维，变为[num_samples, width, height, depth]
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)
        # label 变为整型
        self.train_label = self.train_label.astype(np.int32)  # [60000]
        self.test_label = self.test_label.astype(np.int32)  # [10000]
        # 得到训练和测试的样本数目
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_label[index]


class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # padding='same' 会将周围缺少的部分使用 0 补齐，使得输出的矩阵大小和输入一致。
        self.conv1 = Conv2D(
            filters=32,  # 卷积层神经元数目
            kernel_size=[5, 5],  # 每个卷积核的大小
            padding='same',  # valid: 存在即合理； same: 没有条件创造条件也要上
            activation=tf.nn.relu
        )
        self.pool1 = MaxPool2D(
            pool_size=[2, 2],
            strides=2
        )
        self.conv2 = Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = MaxPool2D(
            pool_size=[2, 2],
            strides=2
        )
        self.flatten = Reshape(target_shape=(7 * 7 * 64,))
        self.dense1 = Dense(units=1024, activation=tf.nn.relu)
        self.dense2 = Dense(units=10)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        output = tf.nn.softmax(x)
        return output


# 迭代epoch
num_epochs = 5
# 学习率
learning_rate = 1e-2
# 批数目
batch_size = 100

model = CNN()
data_loader = MnistLoader()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# 迭代次数
num_batches = int(data_loader.num_train_data // batch_size * num_epochs)

for num_batch in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        print('num_batch: {}, loss: {}'.format(num_batch, loss))

    # 计算梯度
    grads = tape.gradient(loss, model.variables)
    # 更新权重
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

# 输出训练后的模型在测试数据集上的准确率
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
batch_indices = int(data_loader.num_test_data // batch_size)

# 分批次预测图片所属的数字
for batch_index in range(batch_indices):
    start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
    y_pred = model.predict(data_loader.test_data[start_index: end_index])

    # 评估器更新结果
    sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)

# 最后输出测试准确率
print("test accuracy: %f" % sparse_categorical_accuracy.result())
