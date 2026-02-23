Neural Network From Scratch (NumPy)

This project implements a simple 2-layer Neural Network from scratch using NumPy.
It trains on the make_moons dataset and visualizes the learned decision boundary.

No deep learning libraries are used — everything (forward pass, loss, backpropagation, gradient descent) is written manually for learning purposes.

What This Project Demonstrates

Forward propagation

Tanh activation

Softmax for classification

Cross-entropy loss

Backpropagation (manual gradient computation)

L2 regularization

Gradient descent parameter updates

Effect of different hidden layer sizes

Network Architecture

Input (2 features)
→ Hidden Layer (tanh activation)
→ Output Layer (softmax activation)

Output size = number of classes (2).

How It Works

Initialize weights randomly.

Perform forward propagation to compute predictions.

Compute cross-entropy loss.

Use backpropagation to compute gradients.

Update weights using gradient descent.

Repeat for many iterations.

Plot decision boundary to visualize learning.

Requirements
pip install numpy matplotlib scikit-learn
Run
python nn_from_scratch.py
Purpose

This project is designed for beginners who want to understand how neural networks work internally, without using high-level frameworks like TensorFlow or PyTorch.
