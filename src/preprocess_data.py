import pandas as pd
import numpy as np

df = pd.read_csv("./data/raw/train_FD001.txt", sep=" ", header=None)

# name columns
columns = (
    ["unit", "cycle"] +
    [f"op_setting_{i}" for i in range(1, 4)] +
    [f"sensor_{i}" for i in range(1, 24)]
)

df.columns = columns

# drop null sensors
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

X, y = create_sequences(df, sequence_length, features)

print("AI input (seq_nb, seq_length, features_nb):", X.shape)
print("AI target (seq_nb,):",y.shape)