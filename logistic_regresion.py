import numpy as np
import math


def sigmoid(theta, x):
    return 1 / (1 + math.exp(-np.dot(theta, x)))


def log_reg(data, dim): #Returns tuple of Predicting function and cooficient vector theta
    theta = np.zeros(dim + 1)
    alpha = 0.01
    for i in range(1000):
        s = np.zeros(dim + 1)
        for x in data:
            tmp = np.array(x)
            tmp[0] = 1
            s += (sigmoid(tmp, theta) - x[0]) * tmp
        theta = theta - alpha * s

    def Predict(x):
        tmp = np.zeros(dim + 1)
        tmp[0] = 1
        for i in range(dim):
            tmp[i + 1] = x[i]
        return 1 / (1 + math.exp(-np.dot(theta, tmp)))

    return [Predict, theta]


