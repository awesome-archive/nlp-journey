# coding=utf-8
# created by msg on 2019/11/21 5:52 下午

"""
一个简单的线性回归的例子
"""
import tensorflow as tf

# 简单的定义变量和自变量
X = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[1.], [2.]])

# 权重参数初始化
W = tf.Variable(initial_value=[[1.], [2.]])
b = tf.Variable(initial_value=1.)
vars = [W, b]

# 迭代次数
num_epoch = 100

# 随机梯度下降优化器
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2)
for i in range(num_epoch):
    with tf.GradientTape() as tape:
        y_pred = X * W + b
        L = 0.5 * tf.reduce_sum(tf.square(y_pred - y))
        print('{}:{}'.format(i, L))
    # 计算梯度
    grads = tape.gradient(L, vars)
    # 随机梯度下降法计算W和b的更新值
    optimizer.apply_gradients(grads_and_vars=zip(grads, vars))

# 更新后的权重
print(W.numpy(), b.numpy())
