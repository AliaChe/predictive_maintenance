from tensorflow.keras import layers, models, optimizers


def build_model(sequence_length, n_features, config):
    model = models.Sequential([
        layers.LSTM(
            config["model"]["lstm_units_1"],
            input_shape=(sequence_length, n_features),
            return_sequences=True
        ),
        layers.Dropout(config["model"]["dropout"]),

        layers.LSTM(config["model"]["lstm_units_2"]),
        layers.Dropout(config["model"]["dropout"]),

        layers.Dense(config["model"]["dense_units"], activation="relu"),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=optimizers.Adam(
            config["training"]["learning_rate"]
        ),
        loss="mse",
        metrics=["mae"]
    )

    return model