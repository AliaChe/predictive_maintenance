from tensorflow.keras import layers, models, optimizers


def build_model(sequence_length, n_features):
    model = models.Sequential([
        layers.LSTM(32, input_shape=(sequence_length, n_features), return_sequences=True),
        layers.Dropout(0.3),

        layers.LSTM(16),
        layers.Dropout(0.3),

        layers.Dense(16, activation="relu"),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=optimizers.Adam(0.001),
        loss="mse",
        metrics=["mae"]
    )

    return model