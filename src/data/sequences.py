import numpy as np

def create_sequences(df, sequence_length, features):
    X, y = [], []

    for unit in df["unit"].unique():
        unit_df = df[df["unit"] == unit]

        for i in range(len(unit_df) - sequence_length):
            X.append(unit_df[features].iloc[i:i+sequence_length].values)
            y.append(unit_df["RUL"].iloc[i+sequence_length-1])

    return np.array(X), np.array(y)

class TargetScaler:
    def __init__(self, clip_value):
        self.clip_value = clip_value

    def transform(self, y):
        return y / self.clip_value

    def inverse_transform(self, y):
        return y * self.clip_value    