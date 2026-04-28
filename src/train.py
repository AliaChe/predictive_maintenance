from data.prepare_data import load_data, add_rul, split_by_unit
from data.sequences import create_sequences
from tensorflow.keras import callbacks
from models.lstm_model import build_model

df = load_data("./data/raw/train_FD001.txt")
df = add_rul(df)

# set sequence_length
sequence_length = 30

# define features as op_settings and sensors columns
features = [col for col in df.columns if "sensor" in col or "op_setting" in col]

train_df, val_df, test_df = split_by_unit(df)

# create sequences
X_train, y_train = create_sequences(train_df, sequence_length, features)
X_val, y_val     = create_sequences(val_df, sequence_length, features)
X_test, y_test   = create_sequences(test_df, sequence_length, features)

clip_value = 125
y_train = y_train / clip_value
y_val   = y_val / clip_value
y_test  = y_test / clip_value

# create model
sequence_length = X_train.shape[1]
n_features = X_train.shape[2]

model = build_model(sequence_length, n_features)

model.summary()

# add early stopping
callback = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# train model using train and validation data
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=callback,
    verbose=1
)

# evaluate model using test data
test_loss, test_mae = model.evaluate(X_test, y_test)

print("Test MSE:", test_loss)
print("Test MAE:", test_mae)