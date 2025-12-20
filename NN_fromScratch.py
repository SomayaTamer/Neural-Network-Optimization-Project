import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, activation, learning_rate, output_size):

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.output_size = output_size
        self.architecture = [input_size] + hidden_layers + [output_size]

        self.weights = []
        self.biases = []
        self.layer_activations = []

        # Initialize weights and biases for each layer
        for i in range(1, len(self.architecture)):
            prev_size = self.architecture[i - 1]
            curr_size = self.architecture[i]

            # Store activation for this layer
            if i < len(self.architecture) - 1:
                self.layer_activations.append(activation)
            else:
                self.layer_activations.append('softmax')  # Output layer uses softmax

            # Weight initialization based on activation function
            if activation == 'relu':
                std_dev = np.sqrt(2.0 / prev_size)  # He initialization
            elif activation == 'sigmoid' or activation == 'tanh':
                std_dev = np.sqrt(1.0 / prev_size)  # Xavier initialization
            else:
                std_dev = 0.01

            W = np.random.randn(curr_size, prev_size) * std_dev
            b = np.zeros((curr_size, 1))

            self.weights.append(W)
            self.biases.append(b)

    def forward_propagation(self, X):

        # Input preparation: ensure X is (n_features, n_samples)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        else:
            if X.shape[1] == self.input_size:
                X = X.T
            # else: X is already (n_features, n_samples)

        caches = []
        A = X

        # Forward pass
        for l in range(len(self.weights)):
            W = self.weights[l]
            b = self.biases[l]
            activation = self.layer_activations[l]

            A_prev = A

            # Linear transformation: Z = W * A_prev + b
            Z = np.dot(W, A_prev) + b

            # Apply activation function
            if activation == 'relu':
                A = np.maximum(0, Z)
            elif activation == 'sigmoid':
                A = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))  # Clip to avoid overflow
            elif activation == 'tanh':
                A = np.tanh(Z)
            elif activation == 'softmax':
                # Softmax with numerical stability
                exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

            # Cache values for backpropagation
            cache = {
                'Z': Z,
                'A_prev': A_prev,
                'A': A,
                'activation': activation,
            }
            caches.append(cache)

        A_output = A

        return A_output, caches

    def backward_propagation(self, X, y, caches):

        # Ensure y is properly shaped (output_size, n_samples)
        if y.ndim == 1:
            # If y is 1D array of class indices, convert to one-hot
            n_samples = len(y)
            y_onehot = np.zeros((self.output_size, n_samples))
            y_onehot[y.astype(int), np.arange(n_samples)] = 1
            y = y_onehot
        elif y.shape[0] != self.output_size:
            y = y.T  # Transpose to (output_size, n_samples)

        m = y.shape[1]  # Number of samples
        L = len(caches)  # Number of layers
        grads = {'dW': [], 'db': []}

        # ===== OUTPUT LAYER (Softmax + Categorical Cross-Entropy) =====
        last_cache = caches[-1]
        A_output = last_cache['A']  # Softmax output

        # For softmax + cross-entropy: dZ = A - y
        dZ = A_output - y

        A_prev = last_cache['A_prev']

        dW = (1.0 / m) * np.dot(dZ, A_prev.T)
        db = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)

        grads['dW'].insert(0, dW)
        grads['db'].insert(0, db)

        # Gradient flowing to previous layer
        dA_prev = np.dot(self.weights[-1].T, dZ)

        # ===== HIDDEN LAYERS (going backward) =====
        for l in reversed(range(L - 1)):
            cache = caches[l]
            Z = cache['Z']
            A_prev = cache['A_prev']
            A = cache['A']
            activation = cache['activation']

            dA = dA_prev

            # Compute dZ based on activation function derivative
            if activation == 'relu':
                dZ_l = dA * (Z > 0).astype(float)
            elif activation == 'sigmoid':
                dZ_l = dA * A * (1 - A)
            elif activation == 'tanh':
                dZ_l = dA * (1 - A ** 2)
            else:
                dZ_l = dA  # Linear activation

            # Compute gradients
            dW = (1 / m) * np.dot(dZ_l, A_prev.T)
            db = (1 / m) * np.sum(dZ_l, axis=1, keepdims=True)

            # Store gradients
            grads['dW'].insert(0, dW)
            grads['db'].insert(0, db)

            # Compute dA for previous layer
            if l > 0:
                dA_prev = np.dot(self.weights[l].T, dZ_l)

        return grads

    def update_parameters(self, grads):

        for l in range(len(self.weights)):
            self.weights[l] -= self.learning_rate * grads['dW'][l]
            self.biases[l] -= self.learning_rate * grads['db'][l]

    def predict(self, X):

        # Ensure X is properly shaped
        if X.ndim == 1:
            X = X.reshape(1, -1)  # Single sample: (1, n_features)

        # Get probabilities from forward propagation
        probabilities, _ = self.forward_propagation(X)

        # Find the class with highest probability
        predictions = np.argmax(probabilities, axis=0)

        return predictions

    def compute_loss(self, y_pred, y_true):

        # Ensure y_true is properly shaped
        if y_true.ndim == 1:
            n_samples = len(y_true)
            y_onehot = np.zeros((self.output_size, n_samples))
            y_onehot[y_true.astype(int), np.arange(n_samples)] = 1
            y_true = y_onehot
        elif y_true.shape[0] != self.output_size:
            y_true = y_true.T

        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        # Categorical cross-entropy: -mean(sum(y_true * log(y_pred)))
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=0))

        return loss

    def train_step(self, X_batch, y_batch):

        # Forward pass
        y_pred, caches = self.forward_propagation(X_batch)

        # Compute loss
        loss = self.compute_loss(y_pred, y_batch)

        # Backward pass
        grads = self.backward_propagation(X_batch, y_batch, caches)

        # Update parameters
        self.update_parameters(grads)

        return loss

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):

        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        m = X_train.shape[0]

        # Ensure labels are one-hot encoded
        if y_train.ndim == 1 or y_train.shape[0] == 1 or y_train.shape[1] == self.output_size:
            # Convert to one-hot if needed
            if y_train.ndim == 1:
                y_train_labels = y_train
            else:
                y_train_labels = y_train.flatten() if y_train.shape[0] == 1 else np.argmax(y_train, axis=1)

            y_train_onehot = np.zeros((self.output_size, len(y_train_labels)))
            y_train_onehot[y_train_labels.astype(int), np.arange(len(y_train_labels))] = 1
            y_train = y_train_onehot

        if y_val.ndim == 1 or y_val.shape[0] == 1 or y_val.shape[1] == self.output_size:
            if y_val.ndim == 1:
                y_val_labels = y_val
            else:
                y_val_labels = y_val.flatten() if y_val.shape[0] == 1 else np.argmax(y_val, axis=1)

            y_val_onehot = np.zeros((self.output_size, len(y_val_labels)))
            y_val_onehot[y_val_labels.astype(int), np.arange(len(y_val_labels))] = 1
            y_val = y_val_onehot

        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(m)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[:, indices]

            # Mini-batch training
            epoch_loss = 0
            num_batches = max(1, m // batch_size)

            for i in range(num_batches):
                start = i * batch_size
                end = min(start + batch_size, m)

                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[:, start:end]

                loss = self.train_step(X_batch, y_batch)
                epoch_loss += loss

            # Calculate average loss
            avg_loss = epoch_loss / num_batches
            history['train_loss'].append(avg_loss)

            # Validation loss
            y_val_pred, _ = self.forward_propagation(X_val)
            val_loss = self.compute_loss(y_val_pred, y_val)
            history['val_loss'].append(val_loss)

            # Training accuracy
            train_pred = self.predict(X_train)
            train_acc = np.mean(train_pred == np.argmax(y_train, axis=0))
            history['train_acc'].append(train_acc)

            # Validation accuracy
            val_pred = self.predict(X_val)
            val_acc = np.mean(val_pred == np.argmax(y_val, axis=0))
            history['val_acc'].append(val_acc)

            print(
                f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

        return history


# ===== Predict Function for Single Test Sample =====
def predict(test_sample, model, extract_features_func):

    # Extract features from raw sample
    features = extract_features_func(test_sample)

    # Ensure features are in correct shape
    if features.ndim > 1:
        features = features.flatten()

    # Get prediction from model
    predicted_class = model.predict(features.reshape(1, -1))  # Shape: (1, n_features)

    # Extract scalar value
    predicted_class = int(predicted_class[0])

    print(f"Predicted Class: {predicted_class}")

    return predicted_class
