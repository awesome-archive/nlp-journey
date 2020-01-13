import tensorflow as tf

# 定义两个常量
a = tf.constant(1)
b = tf.constant(2)

print(a)
print(b)
print(tf.add(a, b))
print(a + b)

"""
可以看到，都是tf.Tensor类型
tf.Tensor(1, shape=(), dtype=int32)
tf.Tensor(2, shape=(), dtype=int32)
tf.Tensor(3, shape=(), dtype=int32)
tf.Tensor(3, shape=(), dtype=int32)
"""

c = tf.constant([[1, 2], [3, 4]])
d = tf.constant([[5, 6], [7, 8]])

print(tf.matmul(c, d))  # 矩阵乘法

e = tf.constant([1, 2, 3])
f = tf.constant([3, 4, 5])

print(tf.reduce_sum(tf.multiply(e, f)))  # 点乘求和

# 求导数
x = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:  # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
    y = tf.square(x)
    x.assign_add(1.0)
print(x)
y_grad = tape.gradient(y, x)  # 计算y关于x的导数
print([y.numpy(), y_grad.numpy()])

# 求导数
X = tf.constant([[1.0, 2.0], [3.0, 4.0]])
Y = tf.constant([[1.0], [2.0]])

W = tf.Variable(initial_value=[[1.0], [2.0]])
B = tf.Variable(initial_value=1.)

with tf.GradientTape() as tape:
    L = 0.5 * tf.reduce_sum(tf.square(tf.matmul(X, W) + B - Y))

w_grad, b_grad = tape.gradient(L, [W, B])

print(L.numpy(), [w_grad.numpy(), b_grad.numpy()])
