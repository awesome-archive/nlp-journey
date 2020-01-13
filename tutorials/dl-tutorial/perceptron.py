class Perceptron:

    def __init__(self, input_num, activator):
        self.activator = activator
        # 权重向量初始化为0
        self.weights = [0.0 for _ in range(input_num)]
        # 偏置项初始化为0
        self.bias = 0.0

    def __str__(self):
        """
        打印学习到的权重、偏置项
        """
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self, input_x):
        return self.activator(sum([x * w for (x, w) in zip(input_x, self.weights)]) + self.bias)

    def train(self, input_xs, labels, iteration, rate):
        for i in range(iteration):
            self._one_iteration(input_xs, labels, rate)

    def _one_iteration(self, input_xs, labels, learning_rate):
        for (input_x, label) in zip(input_xs, labels):
            output = self.predict(input_x)
            self._update_weights(input_x, output, label, learning_rate)

    def _update_weights(self, input_x, output, label, learning_rate):
        delta = label - output
        self.weights = [w + learning_rate * delta * x for (x, w) in zip(input_x, self.weights)]
        self.bias += learning_rate * delta


def f(x):
    return 1 if x > 0 else 0


def get_training_dataset():
    input_xs = [[1, 1], [0, 0], [1, 0], [0, 1]]
    labels = [1, 0, 0, 0]
    return input_xs, labels


def get_training_dataset_2():
    input_xs = [[1, 1], [0, 0], [1, 0], [0, 1]]
    labels = [1, 0, 1, 1]
    return input_xs, labels


def train_or_perceptron():
    p = Perceptron(2, f)
    input_xs, labels = get_training_dataset_2()
    p.train(input_xs, labels, 10, 0.1)
    return p


def train_and_perceptron():
    p = Perceptron(2, f)
    input_xs, labels = get_training_dataset()
    p.train(input_xs, labels, 10, 0.1)
    return p


if __name__ == '__main__':
    # and
    and_perceptron = train_and_perceptron()
    print(and_perceptron)
    print('1 and 1 = %d' % and_perceptron.predict([1, 1]))
    print('0 and 0 = %d' % and_perceptron.predict([0, 0]))
    print('1 and 0 = %d' % and_perceptron.predict([1, 0]))
    print('0 and 1 = %d' % and_perceptron.predict([0, 1]))

    # or
    or_perceptron = train_or_perceptron()
    print(or_perceptron)
    print('1 or 1 = %d' % and_perceptron.predict([1, 1]))
    print('0 or 0 = %d' % and_perceptron.predict([0, 0]))
    print('1 or 0 = %d' % and_perceptron.predict([1, 0]))
    print('0 or 1 = %d' % and_perceptron.predict([0, 1]))
