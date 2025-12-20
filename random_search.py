import random
from TF_NN import build_library_model, train_library_model

import numpy as np


class RandomSearch:
    def __init__(self, train_features, train_labels, val_features, val_labels, n_iterations=10):
        self.train_features = train_features
        self.train_labels = train_labels
        self.val_features = val_features
        self.val_labels = val_labels
        self.n_iterations = n_iterations

        self.possible_options = {
            'hidden layers': [1, 2, 3, 4, 5],
            'neurons per layer': [32, 64, 128, 256, 512],
            'activation': ['relu', 'tanh', 'sigmoid'],
            'learning rate': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
            'batch size': [16, 32, 64, 128],
            'optimizer': ['SGD', 'Adam', 'RMSProp', 'Adagrad'],
            'epochs': [i for i in range(3, 21)]
        }
        self.history = []
        self.best_config = {}
        self.best_score = 0
        self.input_dim = self.train_features.shape[1]
        self.num_classes = len(np.unique(train_labels))

        random.seed(40)
        np.random.seed(40)

    def sample_hyperparameters(self):
        config = {}
        for name, possibilities in self.possible_options.items():
            config[name] = random.choice(possibilities)
        return config

    def evaluate_hyperparameters(self, hyperparameters):
        # Build the model
        model = build_library_model(self.input_dim, self.num_classes, hyperparameters['hidden layers'],
                                    hyperparameters['neurons per layer'], hyperparameters['activation'],
                                    hyperparameters['learning rate'], hyperparameters['optimizer'])

        # Train the model
        history, trained_model = train_library_model(model, self.train_features, self.train_labels,
                                                     self.val_features, self.val_labels, hyperparameters['epochs'],
                                                     hyperparameters['batch size'])

        accuracy = history.history['val_accuracy'][-1]

        # Store training history
        train_history = {
            'train_loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'train_accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy']
        }

        return accuracy, train_history

    def optimize_hyperparameters(self):
        for i in range(self.n_iterations):
            config = self.sample_hyperparameters()
            score, train_history = self.evaluate_hyperparameters(config)

            self.history.append({
                'config': config,
                'score': score,
            })

            if score > self.best_score:
                self.best_score = score
                self.best_config = config

        return {'best_score': self.best_score, 'best_config': self.best_config, 'history': self.history}

    def get_top_n(self, n=5):
        sorted_history = sorted(self.history, key=lambda x: x['score'], reverse=True)
        return [(h['config'], h['score']) for h in sorted_history[:n]]
