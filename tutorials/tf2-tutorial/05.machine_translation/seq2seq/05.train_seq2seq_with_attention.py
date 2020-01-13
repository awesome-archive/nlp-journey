# coding=utf-8
# created by msg on 2019/11/26 4:19 下午


import re
import os
import time
import unicodedata
import tensorflow as tf
from sklearn.model_selection import train_test_split


# 第一个参数指定字符串标准化的方式。
# NFC表示字符应该是整体组成(比如可能的话就使用单一编码)
# NFD表示字符应该分解为多个组合字符表示。
# unicodedata.category(chr) 把一个字符返回它在UNICODE里分类的类型
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


# 预处理
def preprocess_sentence(w):
    # 转小写换编码
    w = unicode_to_ascii(w.lower().strip())

    # 标点符号前后加上空格
    #  eg: "he is a boy." => "he is a boy ."
    w = re.sub(r"([?.!,¿])", r" \1 ", w)

    # 去掉多余空格
    w = re.sub(r'[" ]+', " ", w)

    # 去掉前后空格
    w = w.rstrip().strip()

    # 加开始和结束字符
    w = '<start> ' + w + ' <end>'
    return w


# 创建数据集
def parse_data(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    sentence_pairs = [line.split('\t') for line in lines]
    preprocessed_sentence_pairs = [(preprocess_sentence(en), preprocess_sentence(spa)) for en, spa in sentence_pairs]
    return zip(*preprocessed_sentence_pairs)


# 分词和词语到索引的转换
def tokenizer(lang):
    # 用空格分词
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=None, filters='', split=' ')
    # 先分词
    lang_tokenizer.fit_on_texts(lang)
    # 再创建词到索引的映射
    tensor = lang_tokenizer.texts_to_sequences(lang)
    # 不够长度的在后边补0
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer


# 获取句子的最大长度
def max_length(tensor):
    return max(len(t) for t in tensor)


# 拆分训练测试数据集
def split(input_tensor, output_tensor, test_size=0.2):
    return train_test_split(input_tensor, output_tensor, test_size=test_size)


# 构建训练集
def make_dataset(input_tensor, output_tensor, batch_size=64, epochs=20, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, output_tensor))
    if shuffle:
        dataset = dataset.shuffle(30000)
    dataset = dataset.repeat(epochs).batch(batch_size, drop_remainder=True)
    return dataset


# 编码器
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_units, encoding_units, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.encoding_units = encoding_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_units)
        self.gru = tf.keras.layers.GRU(self.encoding_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        # before embedding: [batch_size, max_length, embedding_units]
        # after embedding: [batch_size, max_length, encoding_units]
        x = self.embedding(x)
        # output: [batch_size, max_length, encoding_units]
        # state: [batch_size, encoding_units]
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initial_hidden_state(self):
        return tf.zeros((self.batch_size, self.encoding_units))


# attention层
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        # 三个全连接层
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, decoder_hidden, encoder_outputs):
        """
        decoder_hidden: [batch_size, encoding_units]
        encoder_outputs: [batch_size, max_length, encoding_units]
        """
        # decoder_hidden_with_time_axis: [batch_size, 1, encoding_units]
        decoder_hidden_with_time_axis = tf.expand_dims(decoder_hidden, 1)
        # before V: [batch_size, max_length, units]
        # after V: [batch_size, max_length, 1]
        score = self.V(tf.nn.tanh(self.W1(encoder_outputs) + self.W2(decoder_hidden_with_time_axis)))
        # attention_weights: [batch_size, max_length, 1]
        attention_weights = tf.nn.softmax(score, axis=-1)
        # before sum: [batch_size, max_length, encoding_units]
        context_vector = attention_weights * encoder_outputs
        # after sum: [batch_size, encoding_units]
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


# 解码器
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_units, decoding_units, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.decoding_units = decoding_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_units)
        self.gru = tf.keras.layers.GRU(self.decoding_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.decoding_units)

    def call(self, x, hidden, encoding_outputs):
        context_vector, attention_weights = self.attention(hidden, encoding_outputs)
        # before embedding: [batch_size, 1]
        # after embedding: [batch_size, 1, embedding_units]
        x = self.embedding(x)
        # combined_x: [batch_size, 1, embedding_units + encoding_units]
        combined_x = tf.concat([tf.expand_dims(context_vector, axis=1), x], axis=-1)
        # output: [batch_size, 1, decoding_units]
        # state: [batch_size, decoding_units]
        output, state = self.gru(combined_x)
        # output: [batch_size, decoding_units]
        output = tf.reshape(output, (-1, output.shape[2]))
        # output: [batch_size, vocab_size]
        output = self.fc(output)
        return output, state, attention_weights


def train():
    spa_eng_path = 'spa-eng/spa.txt'
    en_dataset, spa_dataset = parse_data(spa_eng_path)
    # 西班牙语到英语的训练
    input_tensor, input_tokenizer = tokenizer(spa_dataset[:30000])
    output_tensor, output_tokenizer = tokenizer(en_dataset[:30000])

    # 拆分数据集为训练集和验证集
    input_train, input_eval, output_train, output_eval = split(input_tensor, output_tensor)
    # 构建tf.data格式
    train_dataset = make_dataset(input_train, output_train)
    eval_dataset = make_dataset(input_eval, output_eval)

    buffer_size = len(input_train)
    embedding_units = 256
    units = 1024
    batch_size = 64
    steps_per_epoch = len(input_train) // batch_size

    input_vocab_size = len(input_tokenizer.word_index) + 1
    output_vocab_size = len(output_tokenizer.word_index) + 1

    # 调用encoder
    encoder = Encoder(input_vocab_size, embedding_units, units, batch_size)
    # 调用解码器
    decoder = Decoder(output_vocab_size, embedding_units, units, batch_size)

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

    # 单步训练
    def train_step(inputs, targets, enc_hidden):
        loss = 0

        # 计算梯度
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(inputs, enc_hidden)
            dec_hidden = enc_hidden
            dec_inputs = tf.expand_dims([output_tokenizer.word_index['<start>']] * batch_size, 1)

            for t in range(1, targets.shape[1]):
                # 得到解码值
                predictions, dec_hidden, _ = decoder(dec_inputs, dec_hidden, enc_output)
                # 每个批次的第t个输出计算损失值
                loss += loss_function(targets[:, t], predictions, loss_object)
                dec_inputs = tf.expand_dims(targets[:, t], 1)
        # 一批数据的损失值
        batch_loss = (loss / int(targets.shape[1]))

        # 参数包括编码器参数和解码器参数
        variables = encoder.trainable_variables + decoder.trainable_variables
        # 计算梯度
        gradients = tape.gradient(loss, variables)
        # 更新权重参数
        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    # 10个epoch
    for epoch in range(10):
        start = time.time()
        enc_hidden = encoder.initial_hidden_state()
        total_loss = 0

        for (batch, (inputs, targets)) in enumerate(train_dataset.take(steps_per_epoch)):
            batch_loss = train_step(inputs, targets, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


def loss_function(real, pred, loss_object):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


if __name__ == '__main__':
    train()
