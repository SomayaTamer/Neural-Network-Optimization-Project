import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

def load_and_preprocess_cifar10():

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

def extract_features(images):
    images = preprocess_input(images * 255.0)
    features = vgg16.predict(images, verbose=0)
    features = features.reshape(features.shape[0], -1)

    return features


