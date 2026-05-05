from data.prepare_data import(
    load_data,
    add_rul,
    split_by_unit,
    scale_data,
    add_rolling_features,
    add_diff_features,
)
from data.sequences import create_sequences, TargetScaler
from tensorflow.keras import callbacks
from models.lstm_model import build_model
from utils.load_config import load_config
from evaluate import plot_predictions, evaluate_units
from sklearn.utils import shuffle

config = load_config()

clip_value = config["data"]["clip_value"]

df = load_data(config["data"]["path"])
df = add_rul(df, clip_value)

# set sequence_length
sequence_length = config["data"]["sequence_length"]

# define features as sensors columns
features = [col for col in df.columns if "sensor" in col]

df = add_rolling_features(df, features, window=5)
df = add_diff_features(df, features)

df.fillna(0, inplace=True)

# add new features (rolling mean and diff on all sensors)
features = [col for col in df.columns if "sensor" in col]

if config["features"]["use_operational_settings"]:
    features += [f"op_setting_{i}" for i in range(1, 4)]

train_df, val_df, test_df = split_by_unit(
    df,
    config["split"]["test_size"],
    config["split"]["val_size"],
    config["split"]["random_state"]
)

train_df, val_df, test_df, scaler = scale_data(
    train_df, val_df, test_df, features
)

# create sequences
X_train, y_train = create_sequences(train_df, sequence_length, features)
X_val, y_val     = create_sequences(val_df, sequence_length, features)
X_test, y_test   = create_sequences(test_df, sequence_length, features)

target_scaler = TargetScaler(clip_value)

y_train = target_scaler.transform(y_train)
y_val   = target_scaler.transform(y_val)
y_test  = target_scaler.transform(y_test)

X_train, y_train = shuffle(X_train, y_train, random_state=config["split"]["random_state"])

# create model
model = build_model(sequence_length, len(features), config)

model.summary()

callback = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=config["training"]["patience"],
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=config["training"]["epochs"],
    batch_size=config["training"]["batch_size"],
    callbacks=[callback]
)

# evaluate model for all tested units or set max_units to evaluate
results = evaluate_units(model, test_df, features, config)

# plot predictions for a given unit
plot_predictions(results, unit_id=3)

# plot a given number of units
plot_predictions(results, n_units=6)