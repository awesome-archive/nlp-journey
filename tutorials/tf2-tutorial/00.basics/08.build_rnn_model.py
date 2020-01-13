# coding=utf-8
# created by msg on 2019/11/22 1:06 下午

import numpy as np
import tensorflow as tf


class DataLoader:
    def __init__(self):
        # 下载 尼采的文章
        path = tf.keras.utils.get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
        with open(path, encoding='utf-8') as f:
            self.raw_text = f.read().lower()
        # 对每个字符简单排序
        self.chars = sorted(list(set(self.raw_text)))
        # 字符到索引的映射
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        # 索引到字符的映射
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        # 替换为索引后的文本
        self.text = [self.char_indices[c] for c in self.raw_text]

    def get_batch(self, seq_length, batch_size):
        seq = []
        next_char = []
        for i in range(batch_size):
            # 随机选择样本
            index = np.random.randint(0, len(self.text) - seq_length)
            # 取固定长度的文本作为输入
            seq.append(self.text[index: index + seq_length])
            # 取固定长度文本之后的文本作为预测的输出
            next_char.append(self.text[index + seq_length])
        return np.array(seq), np.array(next_char)


class RNN(tf.keras.Model):
    def __init__(self, num_chars, batch_size, seq_length):
        super().__init__()
        self.num_chars = num_chars
        self.batch_size = batch_size
        self.seq_length = seq_length

        # 一个lstm的单元
        self.cell = tf.keras.layers.LSTMCell(units=256)
        # 一个全连接层: 输出维度是字符数目
        self.dense = tf.keras.layers.Dense(units=self.num_chars)

    def call(self, inputs, from_logits=False, **kwargs):
        # 把文本变成one hot模式
        inputs = tf.one_hot(inputs, depth=self.num_chars)
        # 初始化状态
        state = self.cell.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
        # 简单的逐个时间步计算输出和状态
        output = None
        for t in range(self.seq_length):
            output, state = self.cell(inputs[:, t, :], state)
        logits = self.dense(output)

        if from_logits:
            return logits
        else:
            return tf.nn.softmax(logits)

    def predict(self, inputs, temperature=1., **kwargs):
        batch_size, _ = tf.shape(inputs)
        logits = self(inputs, from_logits=True)
        prob = tf.nn.softmax(logits / temperature).numpy()
        return np.array([np.random.choice(self.num_chars, p=prob[i, :]) for i in range(batch_size.numpy())])


num_batches = 1000
seq_length = 40
batch_size = 50
learning_rate = 1e-3

data_loader = DataLoader()
model = RNN(num_chars=len(data_loader.chars), batch_size=batch_size, seq_length=seq_length)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
for batch_index in range(num_batches):
    X, y = data_loader.get_batch(seq_length, batch_size)
    with tf.GradientTape() as tape:
        y_pred = model(X)
        # 归结为一个分类问题
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        print('batch: {}, loss: {}'.format(batch_index, loss))
    # 计算损失函数对模型可训练参数的偏导数
    grads = tape.gradient(loss, model.variables)
    # 用梯度更新可训练参数
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

# 取一条数据
X_, _ = data_loader.get_batch(seq_length, 1)
for diversity in [0.2, 0.5, 1.0, 1.2]:
    X = X_
    print('diversity %f: ' % diversity)
    # 预测下一个字符，循环进行下去，就完成了文字的生成，生成400个字符
    for t in range(400):
        y_pred = model.predict(X, diversity, )
        print(data_loader.indices_char[y_pred[0]], end='', flush=True)
        # 去掉第一个字符，加上预测的上一个字符，进行下一轮预测
        X = np.concatenate([X[:, 1:], np.expand_dims(y_pred, axis=1)], axis=-1)
    print('\n')
