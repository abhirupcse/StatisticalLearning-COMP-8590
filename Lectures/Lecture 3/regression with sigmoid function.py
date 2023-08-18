import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generate synthetic regression data using sigmoid function
np.random.seed(42)
x = np.linspace(-10, 10, 100)
y_true = sigmoid(x)
noise = np.random.normal(0, 0.1, size=len(x))
y = y_true + noise

# Plot the synthetic regression data
plt.scatter(x, y, label='Data with Noise')
plt.plot(x, y_true, color='red', label='True Function')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Regression with Sigmoid Function')
plt.legend()
plt.show()
