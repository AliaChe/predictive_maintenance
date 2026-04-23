import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./data/raw/train_FD001.txt", sep=" ", header=None)

# name columns
columns = (
    ["unit", "cycle"] +
    [f"op_setting_{i}" for i in range(1, 4)] +
    [f"sensor_{i}" for i in range(1, 24)]
)

df.columns = columns

print("df description: \n", df.describe())

# drop null sensors
df.dropna(axis=1, inplace=True)
# remaining not nan sensors number
not_nan_sensors_nb = df.columns.size - 5

# choose sensors to study from 1 to max sensors number
sensors =  [f"sensor_{i}" for i in range(19, not_nan_sensors_nb)]

# choose unit (engine) to study
check_unit = 4
unit = df[df["unit"] == check_unit]

# Plot sensors evolution over time for chosen unit
plt.figure(figsize=(10,6))
for s in sensors:
    plt.plot(unit["cycle"], unit[s], label=s)
plt.xlabel("Cycle")
plt.ylabel("Sensor values")
plt.title(f"Sensor evolution - Unit {check_unit}")
plt.legend()

# Plot each given sensor mean evolution over time for all units
for s in sensors:
    plt.figure(figsize=(10,6))
    df.groupby("cycle")[s].mean().plot()
    plt.xlabel("Cycle")
    plt.ylabel("Sensor mean value")
    plt.title(f"{s} mean evolution for all Units")
    plt.legend()
plt.show()

# compute remaining useful life (RUL) for each unit
rul = df.groupby("unit")["cycle"].max().reset_index()
rul.columns = ["unit", "max_cycle"]

df = df.merge(rul, on="unit")
df["RUL"] = df["max_cycle"] - df["cycle"]

print("df with RUL info: \n", df.head())

# update unit dataframe with RUL info
unit = df[df["unit"] == check_unit]

# plot RUL over time for chosen unit
plt.plot(unit["cycle"], unit["RUL"])
plt.title("RUL over time - Unit 1")
plt.xlabel("Cycle")
plt.ylabel("RUL")
plt.show()