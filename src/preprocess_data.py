import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("./data/raw/train_FD001.txt", sep=" ", header=None)

# name columns
columns = (
    ["unit", "cycle"] +
    [f"op_setting_{i}" for i in range(1, 4)] +
    [f"sensor_{i}" for i in range(1, 24)]
)

df.columns = columns

# drop nan sensors (TODO: try less agressive filtering later)
df.dropna(axis=1, inplace=True)

# compute remaining useful life (RUL) for each unit
rul = df.groupby("unit")["cycle"].max().reset_index()
rul.columns = ["unit", "max_cycle"]

df = df.merge(rul, on="unit")
df["RUL"] = df["max_cycle"] - df["cycle"]

# clip RUL max value
df["RUL"] = df["RUL"].clip(upper=125)

# set sequence_length
sequence_length = 30

# define features as op_settings and sensors columns
features = [col for col in df.columns if "sensor" in col or "op_setting" in col]

# create sequences
def create_sequences(df, sequence_length, features):
    X = []
    y = []

    for unit in df["unit"].unique():
        unit_df = df[df["unit"] == unit]

        for i in range(len(unit_df) - sequence_length):
            seq = unit_df[features].iloc[i : i + sequence_length].values
            target = unit_df["RUL"].iloc[i + sequence_length - 1]

            X.append(seq)
            y.append(target)

    return np.array(X), np.array(y)

# split data by unit  into train, test and validation 
units = df["unit"].unique()

train_units, temp_units = train_test_split(
    units, test_size=0.3, random_state=42
)

val_units, test_units = train_test_split(
    temp_units, test_size=0.5, random_state=42
)

train_df = df[df["unit"].isin(train_units)].copy()
val_df   = df[df["unit"].isin(val_units)].copy()
test_df  = df[df["unit"].isin(test_units)].copy()

scaler = StandardScaler()

# fit ONLY on training data
scaler.fit(train_df[features])

# transform all splits
train_df[features] = scaler.transform(train_df[features])
val_df[features]   = scaler.transform(val_df[features])
test_df[features]  = scaler.transform(test_df[features])

# check
print("Train units:", len(train_units))
print("Val units:", len(val_units))
print("Test units:", len(test_units))

print("Train shape:", train_df.shape)
print("Val shape:", val_df.shape)
print("Test shape:", test_df.shape)

# create sequences
X_train, y_train = create_sequences(train_df, sequence_length, features)
X_val, y_val     = create_sequences(val_df, sequence_length, features)
X_test, y_test   = create_sequences(test_df, sequence_length, features)