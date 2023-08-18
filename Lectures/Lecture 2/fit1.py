import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample data
X = np.array([[1], [2], [3], [4], [5]])  # Independent variable
y = np.array([2, 4, 7, 7.8, 10])  # Dependent variable

# Create a linear regression object
regression = LinearRegression()

# Fit the linear regression model
regression.fit(X, y)

# Predict using the model
x_new = np.array([[6]])  # New data point to predict
y_pred = regression.predict(x_new)

# Generate points for plotting the line
x_plot = np.linspace(0, 7, 100).reshape(-1, 1)
y_plot = regression.predict(x_plot)

# Plot the data points and the regression line
plt.scatter(X, y, color='blue', label='Data')
plt.plot(x_plot, y_plot, color='red', label='Regression Line')
plt.scatter(x_new, y_pred, color='green', label='Prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()


# Plot the data and fitted model
sns.regplot(x='experience', y='salary', data=df)
plt.xlabel('Years of Experience')
plt.ylabel('Salary (USD)')
plt.show()


