import numpy as np
import librosa
import os


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

X_data = []
y_data = []

# label = 1
yes_folder = "data/raw/yes"
for file in os.listdir(yes_folder):
    if file.endswith(".wav"):
        path = os.path.join(yes_folder, file)
        features = extract_features(path)
        X_data.append(features)
        y_data.append(1)

# label = 0
no_folder = "data/raw/no"
for file in os.listdir(no_folder):
    if file.endswith(".wav"):
        path = os.path.join(no_folder, file)
        features = extract_features(path)
        X_data.append(features)
        y_data.append(0)

X = np.array(X_data)
y = np.array(y_data)

print(f"Total samples: {len(X)}")

# Train-test split (70-30)
split = int(0.7 * len(X))
X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Normalization
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0) + 1e-8
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# One-hot encoding
def one_hot(y):
    oh = np.zeros((len(y), 2))
    for i in range(len(y)):
        oh[i, y[i]] = 1
    return oh

y_train_oh = one_hot(y_train)
y_test_oh = one_hot(y_test)

# Step 2: Initialize network
input_size = 13
hidden_size = 32
output_size = 2

# Initialize weights and biases
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

print(f"Architecture: {input_size} -> {hidden_size} -> {output_size}")

# Step 3: Define activation functions
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Step 4: Forward propagation
def forward(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# Step 5: Loss function (cross-entropy)
def loss_function(y_true, y_pred):
    m = len(y_true)
    loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
    return loss

# Step 6: Backward propagation
def backward(X, y_true, Z1, A1, A2, W1, W2):
    m = len(X)

    # Output layer
    dZ2 = A2 - y_true
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # Hidden layer
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (Z1 > 0)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2

# Step 7: Training
learning_rate = 0.5
epochs = 300

for epoch in range(epochs):
    # Forward pass
    Z1, A1, Z2, A2 = forward(X_train, W1, b1, W2, b2)

    # Calculate loss
    loss = loss_function(y_train_oh, A2)

    # Calculate accuracy
    predictions = np.argmax(A2, axis=1)
    actual = np.argmax(y_train_oh, axis=1)
    accuracy = np.mean(predictions == actual)

    # Backward pass
    dW1, db1, dW2, db2 = backward(X_train, y_train_oh, Z1, A1, A2, W1, W2)

    # Update parameters (gradient descent)
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    # Print every 20 epochs
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Accuracy: {accuracy*100:.2f}%")

# Step 8: Testing
Z1_test, A1_test, Z2_test, A2_test = forward(X_test, W1, b1, W2, b2)

test_loss = loss_function(y_test_oh, A2_test)
test_pred = np.argmax(A2_test, axis=1)
test_accuracy = np.mean(test_pred == y_test)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

