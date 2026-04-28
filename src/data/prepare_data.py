import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path):
    df = pd.read_csv(path, sep=" ", header=None)

    # name columns
    columns = (
        ["unit", "cycle"] +
        [f"op_setting_{i}" for i in range(1, 4)] +
        [f"sensor_{i}" for i in range(1, 24)]
    )
    df.columns = columns

    # drop nan sensors (TODO: try less agressive filtering later)
    df.dropna(axis=1, inplace=True)

    return df


def add_rul(df, clip_value=125):
    # compute remaining useful life (RUL) for each unit
    rul = df.groupby("unit")["cycle"].max().reset_index()
    rul.columns = ["unit", "max_cycle"]

    df = df.merge(rul, on="unit")
    df["RUL"] = df["max_cycle"] - df["cycle"]

    # clip RUL max value
    df["RUL"] = df["RUL"].clip(upper=clip_value)
    return df


def split_by_unit(df):
    units = df["unit"].unique()

    train_units, temp_units = train_test_split(units, test_size=0.3, random_state=42)
    val_units, test_units = train_test_split(temp_units, test_size=0.5, random_state=42)

    return (
        df[df["unit"].isin(train_units)].copy(),
        df[df["unit"].isin(val_units)].copy(),
        df[df["unit"].isin(test_units)].copy(),
    )


def scale_data(train_df, val_df, test_df, features):
    scaler = StandardScaler()
    # fit ONLY on training data
    scaler.fit(train_df[features])
    
    # transform all splits
    train_df[features] = scaler.transform(train_df[features])
    val_df[features]   = scaler.transform(val_df[features])
    test_df[features]  = scaler.transform(test_df[features])

    return train_df, val_df, test_df, scaler



