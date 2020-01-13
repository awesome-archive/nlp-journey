# coding=utf-8
# created by msg on 2019/11/25 5:39 下午

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# X = tf.constant([2013, 2014, 2015, 2016, 2017])
# Y = tf.constant([12000, 14000, 15000, 16500, 17500])

# 也可以使用NumPy数组，效果相同
X = np.array([2013, 2014, 2015, 2016, 2017])
Y = np.array([12000, 14000, 15000, 16500, 17500])

dataset = tf.data.Dataset.from_tensor_slices((X, Y))

for x, y in dataset:
    print(x.numpy(), '=>', y.numpy())

(train_data, train_label), (_, _) = tf.keras.datasets.mnist.load_data()

train_data = np.expand_dims(train_data.astype(np.float32) / 255.0, axis=-1)

# 从tensor构建dataset
mnist_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))

for image, label in mnist_dataset:
    plt.title(label.numpy())
    plt.imshow(image.numpy()[:, :, 0])
    plt.show()
    break
