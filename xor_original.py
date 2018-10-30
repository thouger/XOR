import tensorflow as tf
import numpy as np

x = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0, 1, 1, 0]])
w1 = np.random.randn(2, 4)
w2 = np.random.randn(4, 1)

LEARNING_RATE = 0.11


# 开始迭代
def sigmoid(y):
    return 1 / (1 + np.exp(-y))


def desigmoid(x):
    return x * (1 - x)


def update():
    global w1, w2
    y_HiddenLayer = sigmoid(np.dot(x, w1))
    y_OutputLayer = sigmoid(np.dot(y_HiddenLayer, w2))

    y_OutputLayer_der = (y_OutputLayer-y.T) * desigmoid(y_OutputLayer)

    y_HiddenLayer_der = y_OutputLayer_der.dot(w2.T) * desigmoid(y_HiddenLayer)

    w2 -= LEARNING_RATE * y_HiddenLayer.T.dot(y_OutputLayer_der)
    w1 -= LEARNING_RATE * x.T.dot(y_HiddenLayer_der)


for i in range(100000):
    update()
    if i % 500 == 5:
        y_HiddenLayer = sigmoid(np.dot(x, w1))
        y_OutputLayer = sigmoid(np.dot(y_HiddenLayer, w2))
        print('Errir:', np.mean(np.abs(y_OutputLayer - y.T)))  # 打印误差

print(y_OutputLayer)

# 显示结果
def judge(x):
    if (x >= 0.5):
        return 1
    else:
        return 0


for i in map(judge, y_OutputLayer):
    print(i)
