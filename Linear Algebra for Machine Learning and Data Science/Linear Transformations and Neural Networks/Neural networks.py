import numpy as np
import pandas as pd
import utils


### FORWARD PROPAGATION ###
def forward_prop(X, parameters):
    W = parameters["W"]
    b = parameters["b"]
    return np.matmul(W, X) + b


### MEAN SQUARED ERROR ###
def mse(yhat, y):
    m = y.shape[1]
    return np.sum((yhat - y) ** 2) / 2 / m


### TRAINING ###
def train(X, y, num_iters=1000, Pcost=False):
    n = X.shape[0]
    parameters = utils.initialize_parameters(n)
    for i in range(num_iters):
        yhat = forward_prop(X, parameters)
        cost = mse(yhat, y)
        parameters = utils.train_nn(parameters, yhat, X, y, learning_rate=0.001)
        if Pcost:
            if i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
    return parameters


### DATASET ###
df = pd.read_csv("data/toy_dataset.csv")
# print(df.head())
# CONVERT TO NUMPY
X = np.array(df[["x1", "x2"]]).T
y = np.array(df["y"]).reshape(1, -1)
# TRAIN MODEL
parameters = train(X, y, num_iters=5000, Pcost=False)


### PREDICTION ###
def predict(X, parameters):
    W = parameters["W"]
    b = parameters["b"]
    return np.matmul(W, X) + b


yhat = predict(X, parameters)
df["y_hat"] = yhat.flatten()
# print(df.head(10))