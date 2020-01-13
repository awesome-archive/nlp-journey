# coding=utf-8
# created by msg on 2019/11/23 5:52 下午

import numpy as np
import tensorflow as tf


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


class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs, training=None, mask=None):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        # softmax 函数能够凸显原始向量中最大的值，并抑制远低于最大值的其他分量，即平滑化的 argmax 函数
        output = tf.nn.softmax(x)

        return output


num_epochs = 50
learning_rate = 1e-2
batch_size = 100
log_dir = 'tensorboard'

model = MLP()
data_loader = MnistLoader()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
# 实例化记录器
summary_writer = tf.summary.create_file_writer(logdir=log_dir)
tf.summary.trace_on(profiler=True)          # 开启Trace（可选）

# 迭代次数
num_batches = int(data_loader.num_train_data // batch_size * num_epochs)

for num_batch in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        print("batch %d: loss %f" % (num_batch, loss.numpy()))
        with summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=num_batch)
    # 计算梯度
    grads = tape.gradient(loss, model.variables)
    # 更新权重
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

with summary_writer.as_default():
    tf.summary.trace_export(name='model_trace', step=0, profiler_outdir=log_dir)
