"""
Simple 2-layer Neural Network (from scratch) on make_moons dataset.
- Pure .py script (no Jupyter magics)
- Logistic Regression part removed
- Trains NN and plots decision boundaries for different hidden sizes
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


# -----------------------------
# Data
# -----------------------------
np.random.seed(3)
X, y = make_moons(n_samples=200, noise=0.20)
num_examples = X.shape[0]  # training set size


# -----------------------------
# Hyperparameters / dimensions
# -----------------------------
nn_input_dim = 2     # input layer dimensionality (2 features)
nn_output_dim = 2    # output layer dimensionality (2 classes)
epsilon = 0.01       # learning rate
reg_lambda = 0.01    # L2 regularization strength


# -----------------------------
# Plot helper
# -----------------------------
def plot_decision_boundary(pred_func, X, y, title=None):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    grid = np.c_[xx.ravel(), yy.ravel()]   # shape: (num_grid_points, 2)
    Z = pred_func(grid)
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors="k")
    if title is not None:
        plt.title(title)


# -----------------------------
# Loss
# -----------------------------
def calculate_loss(model, X, y):
    """
    Cross-entropy loss + L2 regularization (weights only).
    """
    W1, b1, W2, b2 = model["W1"], model["b1"], model["W2"], model["b2"]

    # Forward pass
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2

    # Softmax (with basic numerical stability)
    z2_stable = z2 - np.max(z2, axis=1, keepdims=True)
    exp_scores = np.exp(z2_stable)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Cross-entropy loss
    correct_logprobs = -np.log(probs[np.arange(X.shape[0]), y])
    data_loss = np.sum(correct_logprobs)

    # L2 regularization on weights
    data_loss += (reg_lambda / 2.0) * (np.sum(W1 * W1) + np.sum(W2 * W2))

    # Average loss
    return data_loss / X.shape[0]


# -----------------------------
# Predict
# -----------------------------
def predict(model, x):
    """
    Returns predicted class indices for input x.
    x shape: (m, 2)
    """
    W1, b1, W2, b2 = model["W1"], model["b1"], model["W2"], model["b2"]

    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2

    # Softmax (with basic numerical stability)
    z2_stable = z2 - np.max(z2, axis=1, keepdims=True)
    exp_scores = np.exp(z2_stable)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return np.argmax(probs, axis=1)


# -----------------------------
# Train NN (2-layer)
# -----------------------------
def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    """
    Learns parameters for a 2-layer neural network and returns the model dict.
    - nn_hdim: hidden layer size
    - num_passes: number of gradient descent iterations
    """
    np.random.seed(0)

    # Initialize parameters
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    for i in range(num_passes):
        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2

        # Softmax (with basic numerical stability)
        z2_stable = z2 - np.max(z2, axis=1, keepdims=True)
        exp_scores = np.exp(z2_stable)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation
        delta3 = probs
        delta3[np.arange(num_examples), y] -= 1  # dL/dz2
        dW2 = a1.T.dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)

        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))  # tanh' = 1 - a1^2
        dW1 = X.T.dot(delta2)
        db1 = np.sum(delta2, axis=0, keepdims=True)

        # Regularization (weights only)
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # Gradient descent update
        W1 -= epsilon * dW1
        b1 -= epsilon * db1
        W2 -= epsilon * dW2
        b2 -= epsilon * db2

        # Optionally print the loss
        if print_loss and i % 1000 == 0:
            model_tmp = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
            print(f"Loss after iteration {i}: {calculate_loss(model_tmp, X, y):.6f}")

    # Return trained model
    model = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return model


# -----------------------------
# Main
# -----------------------------
def main():
    # Plot the dataset
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral, edgecolors="k")
    plt.title("Dataset: make_moons")
    plt.show()

    # Train a single model and plot its decision boundary
    model = build_model(X, y, nn_hdim=3, print_loss=True)
    plt.figure()
    plot_decision_boundary(lambda x: predict(model, x), X, y,
                           title="Decision Boundary (hidden layer size = 3)")
    plt.show()

    # Compare multiple hidden layer sizes
    plt.figure(figsize=(16, 32))
    hidden_layer_dimensions = [1, 2, 3, 4, 5, 20, 50]
    for i, nn_hdim in enumerate(hidden_layer_dimensions):
        plt.subplot(5, 2, i + 1)
        model = build_model(X, y, nn_hdim=nn_hdim, num_passes=20000, print_loss=False)
        plot_decision_boundary(lambda x: predict(model, x), X, y,
                               title=f"Hidden Layer size {nn_hdim}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()