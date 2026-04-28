import numpy as np

def create_sequences(df, sequence_length, features):
    X, y = [], []

    for unit in df["unit"].unique():
        unit_df = df[df["unit"] == unit]

        for i in range(len(unit_df) - sequence_length):
            X.append(unit_df[features].iloc[i:i+sequence_length].values)
            y.append(unit_df["RUL"].iloc[i+sequence_length-1])

    return np.array(X), np.array(y)


def scale_targets(y, clip_value):
    return y / clip_value