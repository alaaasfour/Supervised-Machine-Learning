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

