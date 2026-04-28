from data.prepare_data import load_data, add_rul, split_by_unit, scale_data
from data.sequences import create_sequences, scale_targets
from tensorflow.keras import callbacks
from models.lstm_model import build_model
from utils.load_config import load_config

config = load_config()

clip_value = config["data"]["clip_value"]

df = load_data(config["data"]["path"])
df = add_rul(df, clip_value)

# set sequence_length
sequence_length = config["data"]["sequence_length"]

# define features as op_settings and sensors columns
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

y_train = scale_targets(y_train, clip_value)
y_val   = scale_targets(y_val, clip_value)
y_test  = scale_targets(y_test, clip_value)

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

# evaluate model using test data
test_loss, test_mae = model.evaluate(X_test, y_test)

print("Test MSE:", test_loss)
print("Test MAE:", test_mae)