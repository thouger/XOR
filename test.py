import tensorflow as tf
import numpy as np

x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

w1 = np.random.randn(2, 4)
w2 = np.random.randn(4, 1)

LEARNING_RATE = 0.05


# 开始迭代
def sigmoid(y):
    return 1 / (1 + np.exp(-y))

def desigmoid(x):
    return x*(1-x)
def update():
    global w1,w2,y,x
    y_HiddenLayer = sigmoid(np.dot(x , w1))
    y_OutputLayer = sigmoid(np.dot(y_HiddenLayer , w2))

    w2 = y_OutputLayer - LEARNING_RATE * ((y - y_OutputLayer)* * y_HiddenLayer.dot(1 - y_OutputLayer))
    w1 = y_HiddenLayer - LEARNING_RATE * ((y_OutputLayer-y_HiddenLayer)*y_HiddenLayer*(1-y_HiddenLayer))

for i in range(2000):
    update()
    if i %500 == 5:
        y_HiddenLayer = sigmoid(np.dot(x,w1))
        y_OutputLayer = sigmoid(np.dot(y_HiddenLayer,w2))
        print('Errir:', y_OutputLayer - y_HiddenLayer)  # 打印误差

# 显示结果
def judge(x):
    if (x >= 0.5):
        return 1
    else:
        return 0

for i in map(judge, y_OutputLayer):
    print(i)
