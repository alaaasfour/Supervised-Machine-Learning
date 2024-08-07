"""
In this lab, we will automate the process of optimizing 𝑤 and 𝑏 using gradient descent.
"""

import math, copy
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients

# Load the dataset
x_train = np.array([1.0, 2.0]) # Features
y_train = np.array([300.0, 500.0]) # Target Value

# Compute the Cost
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost
    return total_cost


"""
Compute the Gradient Descent for Linear Regression

Arguments:
    x (ndarray (m,)): input data, m examples
    y (ndarray (m,)): target values
    w, b (scalar): model parameters

Returns:
    dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
    dj_db (scalar): The gradient of the cost w.r.t. the parameters b
"""
def compute_gradient(x, y, w, b):
    # Number of training examples
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

plt_gradients(x_train, y_train, compute_cost, compute_gradient)
plt.show()
