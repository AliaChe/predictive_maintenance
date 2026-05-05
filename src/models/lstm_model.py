from tensorflow.keras import layers, models, optimizers
from tensorflow import square, reduce_mean

def weighted_mse(y_true, y_pred):
    weight = 1 + 2 * (1 - y_true)
    return reduce_mean(weight * square(y_true - y_pred))

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
        loss=weighted_mse,
        metrics=["mae"]
    )

    return model