import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

# Generate sample data
np.random.seed(123)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - np.random.rand(16))

# Split data into training and testing sets
X_train, X_test = X[:60], X[60:]
y_train, y_test = y[:60], y[60:]

# Fit the KNN regression model
n_neighbors = 5  # choose the number of neighbors
knn_model = KNeighborsRegressor(n_neighbors=n_neighbors)
knn_model.fit(X_train, y_train)

# Make predictions using the KNN model
y_pred = knn_model.predict(X_test)

# Calculate the mean squared error of the predictions
mse = np.mean((y_pred - y_test) ** 2)
print(f"Mean squared error: {mse:.2f}")

# Plot the results
plt.scatter(X_test, y_test, color='red', label='True')
plt.scatter(X_test, y_pred, color='blue', label='Predicted')
plt.legend()
plt.title(f"KNN Regression (k={n_neighbors}, MSE={mse:.2f})")
plt.xlabel('X')
plt.ylabel('y')
plt.show()
