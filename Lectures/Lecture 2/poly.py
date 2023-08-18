import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Sample data
X = np.array([[1], [2], [3], [4], [5]]) # Independent variable
y = np.array([2, 4.7, 6.8, 8, 10]) # Dependent variable

# Transform features into polynomial features
poly_transform = PolynomialFeatures(degree=2) # Set the degree of the polynomial
X_poly = poly_transform.fit_transform(X)

# Create a linear regression object
regression = LinearRegression()

# Fit the linear regression model using the polynomial features
regression.fit(X_poly, y)

# Generate points for plotting the polynomial curve
x_plot = np.linspace(0, 7, 100).reshape(-1, 1)
x_plot_poly = poly_transform.transform(x_plot)
y_plot = regression.predict(x_plot_poly)

# Make predictions on new data points
X_new = np.array([[6], [7]])
X_new_poly = poly_transform.transform(X_new)
y_new = regression.predict(X_new_poly)

# Print the predicted values for new data points
print("Predicted values for new data points:")
for i in range(len(X_new)):
    print("X =", X_new[i][0], "  Predicted y =", y_new[i])

# Plot the data points, polynomial curve, and predicted values
plt.scatter(X, y, color='blue', label='Data')
plt.plot(x_plot, y_plot, color='red', label='Polynomial Regression')
plt.scatter(X_new, y_new, color='green', label='Predicted Values')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression')
plt.legend()
plt.show()

