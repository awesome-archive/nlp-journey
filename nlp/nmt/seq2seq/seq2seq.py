# coding=utf-8
# created by msg on 2019/12/6 4:43 下午

import tensorflow as tf


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
