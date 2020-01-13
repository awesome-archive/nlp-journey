from .perceptron import Perceptron


def f(x):
    return x


class LinearUnit(Perceptron):

    def __init__(self, input_num):
        Perceptron.__init__(self, input_num, f)


# 构造简单的数据集
def get_training_dataset():
    # 构建数据集，输入向量列表，每一项是工作年限
    input_xs = [[5], [3], [8], [1.4], [10.1]]
    # 期望的输出列表，也就是输入向量的对应的标签，与工作年限对应的收入年薪
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_xs, labels


# 使用我们的训练数据集对线性单元进行训练
def train_linear_unit():
    # 创建感知器对象，输入参数的个数也就是特征数为 1（工作年限）
    lu = LinearUnit(1)
    # 获取构建的数据集
    input_xs, labels = get_training_dataset()
    # 训练感知器，迭代 10 轮，学习率为 0.01
    lu.train(input_xs, labels, 10, 0.01)
    # 返回训练好的线性单元
    return lu


# 将图像画出来
def plot(model):
    # 引入绘图的库
    import matplotlib.pyplot as plt
    # 获取训练数据：特征 input_xs 与 对应的标签 labels
    input_xs, labels = get_training_dataset()
    # figure() 创建一个 Figure 对象，与用户交互的整个窗口，这个 figure 中容纳着 subplots
    fig = plt.figure()
    # 在 figure 对象中创建 1行1列中的第一个图
    ax = fig.add_subplot(111)
    # scatter(x, y) 绘制散点图，其中的 x,y 是相同长度的数组序列

    ax.scatter(list(map(lambda x: x[0], input_xs)), labels)

    # 设置权重
    weights = model.weights
    # 设置偏置项
    bias = model.bias

    y1 = 0 * weights[0] + bias
    y2 = 12 * weights[0] + bias
    # 将图画出来
    plt.plot([0, 12], [y1, y2])

    # 将最终的图展示出来
    plt.show()


if __name__ == '__main__':
    # 首先训练我们的线性单元
    linear_unit = train_linear_unit()
    # 打印训练获得的权重 和 偏置
    print(linear_unit)
    # 测试
    print('Work 3.4 years, monthly salary = %.2f' % linear_unit.predict([3.4]))
    print('Work 15 years, monthly salary = %.2f' % linear_unit.predict([15]))
    print('Work 1.5 years, monthly salary = %.2f' % linear_unit.predict([1.5]))
    print('Work 6.3 years, monthly salary = %.2f' % linear_unit.predict([6.3]))
    plot(linear_unit)
