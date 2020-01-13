# coding=utf-8
# created by msg on 2019/12/2 11:21 下午

import tensorflow as tf


def build_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=16, input_shape=(10,), activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='sgd',
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC()]
                  )
    model.summary()


build_model()
