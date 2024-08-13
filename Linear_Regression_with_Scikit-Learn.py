"""
In this lab, we will use the scikit-learn library to perform the linear regression.
This toolkit contains implementation of many useful machine learning functions.
"""

# Packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from lab_utils_multi import load_house_data
from lab_utils_common import dlc
np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')

"""
Gradient Descent
Scikit-learn has a gradient descent regression model `sklearn.linear_model.SGDRegressor`. This model performs best with normalized inputs. 
`sklearn.preprocessing.StandardScaler` will perform z-score normalization. Here it is referred to as 'standard score'.
"""
# Load the dataset
X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']

# Scale/normalize the training data
scalar = StandardScaler()
X_norm = scalar.fit_transform(X_train)
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

# Create and fit the regression model
sgd_reg = SGDRegressor(max_iter = 1000)
sgd_reg.fit(X_norm, y_train)
print(sgd_reg)
print(f"number of iterations completed: {sgd_reg.n_iter_}, number of weight updates: {sgd_reg.t_}")

# View parameters
b_norm = sgd_reg.intercept_
w_norm = sgd_reg.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")
print( "model parameters from previous lab: w: [110.56 -21.27 -32.71 -37.97], b: 363.16")

# Make predictions
# make a prediction using sgd_reg.predict()
y_pred_sgd = sgd_reg.predict(X_norm)
# make a prediction using w,b.
y_pred = np.dot(X_norm, w_norm) + b_norm
print(f"Prediction using np.dot() and sgd_reg.predict match: {(y_pred == y_pred_sgd).all()}")

print(f"Prediction on training set:\n{y_pred[:4]}" )
print(f"Target values \n{y_train[:4]}")

# Plot the results
# plot predictions and targets vs original features
fig, ax = plt.subplots(1,4,figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],y_pred,color=dlc["dlorange"], label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("Target versus prediction using z-score normalized model")
plt.show()