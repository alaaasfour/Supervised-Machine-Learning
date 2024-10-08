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


"""
Feature Scaling 
z-score normalization: After z-score normalization, all features will have a mean of 0 and a standard deviation of 1.

Args:
    X (ndarray (m,n))     : input data, m examples, n features
      
Returns:
    X_norm (ndarray (m,n)): input normalized by column
    mu (ndarray (n,))     : mean of each feature
    sigma (ndarray (n,))  : standard deviation of each feature
"""
def zscore_normalize_feature(X):
    # Find the mean of each column/feature
    mu = np.mean(X, axis=0)

    # Find the standard deviation of each column/feature
    sigma = np.std(X, axis=0)

    # Element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma

    return (X_norm, mu, sigma)

mu     = np.mean(X_train,axis=0)
sigma  = np.std(X_train,axis=0)
X_mean = (X_train - mu)
X_norm = (X_train - mu)/sigma

fig,ax=plt.subplots(1, 3, figsize=(12, 3))
ax[0].scatter(X_train[:,0], X_train[:,3])
ax[0].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[0].set_title("unnormalized")
ax[0].axis('equal')

ax[1].scatter(X_mean[:,0], X_mean[:,3])
ax[1].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[1].set_title(r"X - $\mu$")
ax[1].axis('equal')

ax[2].scatter(X_norm[:,0], X_norm[:,3])
ax[2].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[2].set_title(r"Z-score normalized")
ax[2].axis('equal')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle("distribution of features before, during, after normalization")
plt.show()

# Now, let's normalize the data and compare it to the original data
# Normalize the original feature
X_norm, X_mu, X_sigma = zscore_normalize_feature(X_train)
print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

fig,ax=plt.subplots(1, 4, figsize=(12, 3))
for i in range(len(ax)):
    norm_plot(ax[i],X_train[:,i],)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("count");
fig.suptitle("distribution of features before normalization")
plt.show()
fig,ax=plt.subplots(1,4,figsize=(12,3))
for i in range(len(ax)):
    norm_plot(ax[i],X_norm[:,i],)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("count");
fig.suptitle("distribution of features after normalization")

plt.show()

# Let's rerun the gradient descent algorithm with normalized data.
w_norm, b_norm, hist = run_gradient_descent(X_norm, y_train, 1000, 1.0e-1, )

#Now, let's predict target using normalized features
m = X_norm.shape[0]
yp = np.zeros(m)
for i in range(m):
    yp[i] = np.dot(X_norm[i], w_norm) + b_norm

    # plot predictions and targets versus original features
fig,ax=plt.subplots(1,4,figsize=(12, 3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],yp,color=dlc["dlorange"], label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()


"""
The point of generating the model is to use it to predict housing prices that are not in the data set. 
Let's predict the price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old. 
Recall, that you must normalize the data with the mean and standard deviation derived when the training data was normalized.
"""

# First, normalize out example
x_house = np.array([1200, 3, 1, 40])
x_house_norm = (x_house - X_mu) / X_sigma
print(x_house_norm)
x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.0f}")

plt_equal_scale(X_train, X_norm, y_train)