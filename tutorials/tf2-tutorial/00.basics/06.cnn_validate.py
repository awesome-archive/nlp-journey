# coding=utf-8
# created by msg on 2019/11/22 10:18 上午

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

"""
简单验证一下cnn里的valid和same参数
"""

# 两维数组，再添加一维代表灰度图片
image = np.array([[
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 2, 1, 0],
    [0, 0, 2, 2, 0, 1, 0],
    [0, 1, 1, 0, 2, 1, 0],
    [0, 0, 2, 1, 1, 0, 0],
    [0, 2, 1, 1, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
]], dtype=np.float32)

plt.imshow(image[0])
plt.show()

image = np.expand_dims(image, axis=-1)

print(image.shape)

W = np.array([[
    [0, 0, -1],
    [0, 1, 0],
    [-2, 0, 2]
]], dtype=np.float32)

b = np.array([1], dtype=np.float32)

# valid 输出多少维算多少维，same要保持与输入相同的维度
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=[2, 2],
        padding='same',  # 'valid'
        activation=tf.nn.relu
    )
])

output = model(image)

# 输出查看shape，可知 same保持了与输入相同的shape, valid则不会
print(tf.squeeze(output))
print(output.shape)
