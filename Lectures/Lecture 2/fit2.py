import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Create some sample data with two features
X = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
y = np.array([5, 7, 9, 11, 13])

# Initialize a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Print the model coefficients
print("Coefficients:", model.coef_)

# Predict the output for new data points
X_new = np.array([[6, 12], [7, 14], [8, 16]])
y_pred = model.predict(X_new)
print("Predictions:", y_pred)

# Plot the data and the linear regression plane in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], y)

# Create a meshgrid to plot the regression plane
xx, yy = np.meshgrid(range(10), range(10))
zz = model.coef_[0] * xx + model.coef_[1] * yy + model.intercept_

# Plot the regression plane
ax.plot_surface(xx, yy, zz, alpha=0.5)

# Add axis labels and a title to the plot
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target')
plt.title('Linear Regression with Two Features in 3D')

plt.show()
