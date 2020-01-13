# coding=utf-8
# created by msg on 2019/12/6 4:44 下午

import os
import time
from nlp.nmt.seq2seq.seq2seq import *
from nlp.nmt.seq2seq.preprocess import *


def loss_function(real, pred, loss_object):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


def train():
    spa_eng_path = '../../../demos/nmt/spa-eng/spa.txt'
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


if __name__ == '__main__':
    train()
