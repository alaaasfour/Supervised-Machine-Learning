"""
In this lab, we will achieve the following items:
    - Utilize the multiple variables routines developed in the previous lab
    - Run Gradient Descent on a data set with multiple features
    - Explore the impact of the learning rate alpha on gradient descent
    - Improve performance of gradient descent by feature scaling using z-score normalization
"""

# Packages
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)
from lab_utils_multi import load_house_data, run_gradient_descent, norm_plot, plt_equal_scale, plot_cost_i_w
from lab_utils_common import dlc
plt.style.use('./deeplearning.mplstyle')

"""
The dataset:
We will use the motivating example of housing price prediction. The training data set contains many examples with 4 features 
(size, bedrooms, floors and age). In this lab, the Size feature is in sqft. This data set is large.
We would like to build a linear regression model using these values so we can then predict the price for 
other houses - say, a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old.
"""

# Load the dataset
X_train, y_train = load_house_data()
X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']

# Let's view the dataset and its features by plotting each feature versus price
fig, ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i], y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price (1000's)")
plt.show()

# Learning Rate (alpha)
# Set alpha to 9.9e-7
_,_, hist = run_gradient_descent(X_train, y_train, 10, 9.9e-7)
plot_cost_i_w(X_train, y_train, hist)


# Set alpha to 9e-7
_,_,hist = run_gradient_descent(X_train, y_train, 10, alpha = 9e-7)
plot_cost_i_w(X_train, y_train, hist)

# Set alpha to 1e-7
_,_,hist = run_gradient_descent(X_train, y_train, 10, alpha = 1e-7)
plot_cost_i_w(X_train, y_train, hist)


