import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./NASA_data/train_FD001.txt", sep=" ", header=None)

columns = (
    ["unit", "cycle"] +
    [f"op_setting_{i}" for i in range(1, 4)] +
    [f"sensor_{i}" for i in range(1, 24)]
)

df.columns = columns

df.dropna(axis=1)

sensors =  [f"sensor_{i}" for i in range(19, 24)]

check_unit = 4
unit = df[df["unit"] == check_unit]

plt.figure(figsize=(10,6))
for s in sensors:
    plt.plot(unit["cycle"], unit[s], label=s)
plt.xlabel("Cycle")
plt.ylabel("Sensor values")
plt.title(f"Sensor evolution - Unit {check_unit}")
plt.legend()
plt.show()
    
for s in sensors:
    plt.figure(figsize=(10,6))
    df.groupby("cycle")[s].mean().plot()
plt.show()

rul = df.groupby("unit")["cycle"].max().reset_index()
rul.columns = ["unit", "max_cycle"]

df = df.merge(rul, on="unit")
df["RUL"] = df["max_cycle"] - df["cycle"]

print(df.head())