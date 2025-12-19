import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def load_mask_dataset(dataset_path, img_size=(64, 64)):
    X = []
    y = []

    # Class mapping
    class_map = {
        "with_mask": 1,
        "without_mask": 0
    }

    # Loop through folders
    for class_name, label in class_map.items():
        class_folder = os.path.join(dataset_path, class_name)

        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)

            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(img_size)

                X.append(np.array(img).flatten())
                y.append(label)

            except:
                pass  # ignore corrupt images

    X = np.array(X, dtype="float32") / 255.0
    y = np.array(y)

    return X, y


# Load dataset
dataset_path = "dataset"  # <-- change to your folder name
X, y = load_mask_dataset(dataset_path)

# Split into train / validation / test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)


class NEURALNETWORK:
    def __init__(self, layerSizes, activationFunction):
        self.layerSizes = layerSizes
        self.activationFunction = activationFunction
        self.parameters = self.InitializeParameters()

    def InitializeParameters(self):
        parameters = {}
        for i in range(1, len(self.layerSizes)):
            parameters[f"W{i}"] = np.random.randn(self.layerSizes[i], self.layerSizes[i - 1]) * 0.01
            parameters[f"b{i}"] = np.zeros((self.layerSizes[i], 1))
        return parameters

    def Sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def Relu(self, x):
        return np.maximum(0, x)

    def Forward(self, x):
        self.cache = {}

        # store input as "A0" for backward
        self.cache["A0"] = x.T
        output = x.T
        layersNum = len(self.parameters) // 2

        for i in range(1, layersNum):
            summation = self.parameters[f"W{i}"] @ output + self.parameters[f"b{i}"]
            self.cache[f"Z{i}"] = summation  # use Z for derivatives
            if (self.activationFunction == "sigmoid"):
                output = self.Sigmoid(summation)
            else:
                output = self.Relu(summation)
            self.cache[f"A{i}"] = output 

        finalSummation = self.parameters[f"W{layersNum}"] @ output + self.parameters[f"b{layersNum}"]
        finalOutput = self.Sigmoid(finalSummation)
        self.cache[f"Z{layersNum}"] = finalSummation
        self.cache[f"A{layersNum}"] = finalOutput

        return finalOutput

    def dSigmoid(self, Z):
        s = self.Sigmoid(Z)
        return s * (1 - s)   

    def dRelu(self, Z):
        return (Z > 0).astype(float)

    def Backward(self, y, alpha):
        m = y.shape[0]  # number of examples
        y = y.reshape(1, m)
        grads = {}
        L = len(self.parameters) // 2  # number of layers

        # Output layer
        # Δwji​=η(tj​−oj​)(oj​(1−oj​))oi​
        output = self.cache[f"A{L}"] # output oj
        error = output - y
        grads[f"dW{L}"] = (error @ self.cache[f"A{L-1}"].T) / m  # previous layer oi
        grads[f"db{L}"] = np.sum(error, axis=1, keepdims=True) / m

        # Hidden layer
        # Δwji​=η ∑​wkj(δk​) (oj​(1−oj​)) oi​
        for i in range(L - 1, 0, -1):
            dA_prev = self.parameters[f"W{i+1}"].T @ error #summation 

            if self.activationFunction == "sigmoid":
                error = dA_prev * self.dSigmoid(self.cache[f"Z{i}"])
            else:
                error = dA_prev * self.dRelu(self.cache[f"Z{i}"])

            grads[f"dW{i}"] = (error @ self.cache[f"A{i-1}"].T) / m # oi previous layer output
            grads[f"db{i}"] = np.sum(error, axis=1, keepdims=True) / m

        #  update parameters 
        for i in range(1, L + 1):
            self.parameters[f"W{i}"] -= alpha * grads[f"dW{i}"] # weight update ->  w=w-ηΔw
            self.parameters[f"b{i}"] -= alpha * grads[f"db{i}"] # bias update -> b=b-ηΔb

        return grads

    def Loss(self, y, y_pred):
        m = y.shape[0]
        y = y.reshape(1, m)

        eps = 1e-8  # to avoid log(0)
        loss = -np.mean(
            y * np.log(y_pred + eps) +
            (1 - y) * np.log(1 - y_pred + eps)
        )

        return loss

    def predict(self, X, threshold=0.5):
        y_pred = self.Forward(X)  # shape (1, m)
        predictions = (y_pred >= threshold).astype(int)
        return predictions.flatten()
