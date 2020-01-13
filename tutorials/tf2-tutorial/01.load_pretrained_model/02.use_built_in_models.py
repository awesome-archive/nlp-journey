# coding=utf-8
# created by msg on 2019/11/21 9:45 下午

import tensorflow as tf
import tensorflow_datasets as tfds

# 一些预训练好的模型
"""
pip install tensorflow_datasets

model = tf.keras.applications.MobileNetV2()
model1 = tf.keras.applications.VGG16()
model2 = tf.keras.applications.ResNet50V2()
model3 = tf.keras.applications.ResNet101V2()
model4 = tf.keras.applications.ResNet152V2()
model5 = tf.keras.applications.VGG19()
model6 = tf.keras.applications.NASNetMobile()
"""

num_batches = 1000
batch_size = 50
learning_rate = 0.001

dataset = tfds.load('tf_flowers', split=tfds.Split.TRAIN, as_supervised=True)
dataset = dataset.map(lambda image, label: (tf.image.resize(image, [224, 224]) / 255., label)).shuffle(1024).batch(32)

model = tf.keras.applications.MobileNetV2(weights=None, classes=5)

# 用MobileNetV2模型训练，不使用已经训练好的权重
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
for images, labels in dataset:
    # 可控性比较好的梯度计算方式
    with tf.GradientTape() as tape:
        output = model(images)
        # 稀疏类别的交叉熵算法：类别可以是单独的值，如果是categorical_crossentropy，则类别是one_hot形式
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=labels, y_pred=output)
        # 对损失值求平均
        loss = tf.reduce_mean(loss)
        print('loss:{}'.format(loss))

    # 计算损失函数对模型可训练参数的偏导数
    grads = tape.gradient(loss, model.trainable_variables)
    # 用梯度更新可训练参数
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))

print(model.trainable_variables)
