from tensorflow.keras import layers, models, optimizers

def build_library_model(input_dim=512, num_classes=10, hidden_layers=2, neurons_per_layer=128, activation='relu', learning_rate=0.001, optimizer_type='Adam'):

    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for i in range(hidden_layers):
        model.add(layers.Dense(neurons_per_layer, activation=activation))

    model.add(layers.Dense(num_classes, activation='softmax'))

    opt_dict = {
        'Adam': optimizers.Adam(learning_rate=learning_rate),
        'SGD': optimizers.SGD(learning_rate=learning_rate),
        'RMSProp': optimizers.RMSprop(learning_rate=learning_rate),
        'Adagrad': optimizers.Adagrad(learning_rate=learning_rate)
    }

    selected_optimizer = opt_dict.get(optimizer_type, optimizers.Adam(learning_rate=learning_rate))

    model.compile(
        optimizer = selected_optimizer,
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )
    return model


def train_library_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):

    history = model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs = epochs,
        batch_size = batch_size,
    )
    return history, model
