import numpy as np
import pandas as pd
import sys

class LogisticRegressionClassifier():
    def __init__(self, x_shape):
        self.init_parameters = {}
        self.init_parameters["weight"] = np.zeros(x_shape)
        self.init_parameters["bias"] = 0

    def sigmoid(self, input):
        output = 1 / (1 + np.exp(-input))
        return output

    def optimize(self, x, y,learning_rate,iterations):
        size = x.shape[0]
        weight = self.init_parameters["weight"]
        bias = self.init_parameters["bias"]
        for i in range(iterations):
                sigma = self.sigmoid(np.dot(x.astype(np.float16), weight.astype(np.float16)) + bias)
                loss = -1/size * np.sum(y * np.log(sigma)) + (1 - y) * np.log(1-sigma)
                dW = 1/size * np.dot(x.T, (sigma - y))
                db = 1/size * np.sum(sigma - y)
                weight -= learning_rate * dW
                bias -= learning_rate * db

        self.init_parameters["weight"] = weight
        self.init_parameters["bias"] = bias
        return self.init_parameters

    def fit(self, x, y, learning_rate, iterations):
        parameters_out = self.optimize(x, y, learning_rate, iterations)
        return parameters_out

    def predict(self, X):
        output_values = np.dot(X, self.init_parameters["weight"]) + self.init_parameters["bias"]
        labels = self.sigmoid(output_values)
        return labels
