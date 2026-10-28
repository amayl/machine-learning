import numpy as np
from matplotlib import pyplot as plt

'''
Initialise: W1, W2, W3
           b1, b2, b3
'''

# Apple (AAPL) closing prices, USD
# Each row contains 5 consecutive trading sessions.

X = np.array([
    [336.67, 340.42, 336.64, 340.97, 341.27],
    [340.42, 336.64, 340.97, 341.27, 342.96],
    [336.64, 340.97, 341.27, 342.96, 342.00],
    [340.97, 341.27, 342.96, 342.00, 343.46],
    [341.27, 342.96, 342.00, 343.46, 342.94],
    [342.96, 342.00, 343.46, 342.94, 339.50],
    [342.00, 343.46, 342.94, 339.50, 343.99],
    [343.46, 342.94, 339.50, 343.99, 348.22],
    [342.94, 339.50, 343.99, 348.22, 347.73]
])

# Next trading session's closing price for each row
y = np.array([
    [342.96],
    [342.00],
    [343.46],
    [342.94],
    [339.50],
    [343.99],
    [348.22],
    [347.73],
    [343.91]
])

X_mean = np.mean(X)
X_std = np.std(X)

y_mean = np.mean(y)
y_std = np.std(y)

X_scaled = (X - X_mean) / X_std
y_scaled = (y - y_mean) / y_std

def ReLU(z):
    return np.maximum(0,z)

np.random.seed(42)

# Layer 1: 5 inputs -> 8 neurons
W1 = np.random.randn(5, 8) * np.sqrt(2 / 5)
b1 = np.zeros((1, 8))

# Layer 2: 8 neurons -> 4 neurons
W2 = np.random.randn(8, 4) * np.sqrt(2 / 8)
b2 = np.zeros((1, 4))

# Layer 3: 4 neurons -> 1 output
W3 = np.random.randn(4, 1) * np.sqrt(1 / 4)
b3 = np.zeros((1, 1))

learning_rate = 0.001

def forward_pass(X, W1, b1, W2, b2, W3, b3):
    # first layer
    z1 = X @ W1 + b1
    a1 = ReLU(z1)

    # second layer
    z2 = a1 @ W2 + b2
    a2 = ReLU(z2)

    # output layer
    y_hat = a2 @ W3 + b3
    return y_hat, z1, a1, z2, a2

# wtf is scikit
def mean_squared_error(y, y_hat):
    if len(y) != len(y_hat):
        raise ValueError("Lists must be of same length")

    total_squared_error = 0
    n = len(y)
    for i in range(n):
        diff = y[i] - y_hat[i]
        total_squared_error += diff**2

    return total_squared_error / n

def backpropagation(X, y, y_hat, z1, a1, z2, a2,
                    W1, b1, W2, b2, W3, b3, learning_rate):

    m = X.shape[0]

    # Output layer gradients
    d_y_hat = (2 / m) * (y_hat - y)

    dW3 = a2.T @ d_y_hat
    db3 = np.sum(d_y_hat, axis=0, keepdims=True)

    # Second hidden layer gradients
    dA2 = d_y_hat @ W3.T
    dZ2 = dA2 * (z2 > 0)

    dW2 = a1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    # First hidden layer gradients
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (z1 > 0)

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Update weights and biases
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3

    return W1, b1, W2, b2, W3, b3

epochs = 5100
loss_history = []

for epoch in range(epochs):

    # 1. Forward pass
    y_hat, z1, a1, z2, a2 = forward_pass(
    X_scaled, W1, b1, W2, b2, W3, b3
    )

    # loss function
    loss = mean_squared_error(y_scaled, y_hat)
    loss_history.append(loss)

    # backpropogation
    W1, b1, W2, b2, W3, b3 = backpropagation(
        X_scaled, y_scaled, y_hat, z1, a1, z2, a2,
        W1, b1, W2, b2, W3, b3, learning_rate
    )

    # 4. Print the loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Plot training loss and predictions in the original price scale.
y_pred = y_hat * y_std + y_mean

fig, (loss_ax, price_ax) = plt.subplots(1, 2, figsize=(12, 4))

loss_ax.plot(loss_history)
loss_ax.set_title("Training loss")
loss_ax.set_xlabel("Epoch")
loss_ax.set_ylabel("Mean squared error")
loss_ax.grid(True)

price_ax.plot(y, marker="o", label="Actual")
price_ax.plot(y_pred, marker="x", label="Predicted")
price_ax.set_title("Apple closing prices")
price_ax.set_xlabel("Sample")
price_ax.set_ylabel("Price (USD)")
price_ax.legend()
price_ax.grid(True)

plt.tight_layout()
plt.show()

