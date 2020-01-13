# coding=utf-8
# created by msg on 2019/11/24 5:03 下午

import re
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


if __name__ == '__main__':
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

    for x, y in train_dataset.take(1):
        pass

    embedding_units = 256
    units = 1024
    batch_size = 64
    input_vocab_size = len(input_tokenizer.word_index) + 1
    output_vocab_size = len(output_tokenizer.word_index) + 1

    # 调用encoder
    encoder = Encoder(input_vocab_size, embedding_units, units, batch_size)
    sample_hidden_state = encoder.initial_hidden_state()
    sample_output, sample_hidden_state = encoder(x, sample_hidden_state)

    print(sample_output.shape)
    print(sample_hidden_state.shape)


