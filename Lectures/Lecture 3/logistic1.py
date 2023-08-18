import numpy as np
from scipy.optimize import minimize

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def negative_log_likelihood(params, X, y):
    beta_0, beta_1 = params
    z = beta_0 + beta_1 * X
    p = sigmoid(z)
    loss = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return loss

def predict(params, X):
    beta_0, beta_1 = params
    z = beta_0 + beta_1 * X
    p = sigmoid(z)
    return p

# Generate synthetic data for binary classification
np.random.seed(42)
X = np.random.randn(100)
y = np.random.choice([0, 1], size=100)

# Define the initial parameters
initial_params = np.zeros(2)

# Minimize the negative log-likelihood loss function
result = minimize(negative_log_likelihood, initial_params, args=(X, y), method='BFGS')
optimal_params = result.x

print("Optimal Parameters:", optimal_params)

# Predict new values
new_X = np.array([-1, 0, 1])
predictions = predict(optimal_params, new_X)

print("New Data Predictions:", predictions)
