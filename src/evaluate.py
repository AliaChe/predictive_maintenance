import matplotlib.pyplot as plt
import numpy as np
import math
from data.sequences import create_sequences, TargetScaler

def plot_predictions(results, unit_id=None, n_units=None, max_cols=3):

    unit_ids = list(results.keys())

    # ---------------------------
    # CASE 1: single unit
    # ---------------------------
    if unit_id is not None:
        if unit_id not in results:
            print(f"Warning: Unit {unit_id} not found in results")
            return

        data = results[unit_id]

        plt.figure(figsize=(8, 6))
        plt.plot(data["y_true"], label="True RUL")
        plt.plot(data["y_pred"], label="Predicted RUL")
        plt.legend()
        plt.title(f"RUL over time - Unit {unit_id}")
        plt.show()
        return

    # ---------------------------
    # CASE 2: select subset of units
    # ---------------------------
    if n_units is not None:
        unit_ids = unit_ids[:n_units]

    n_units_total = len(unit_ids)

    # grid layout
    n_cols = max_cols
    n_rows = math.ceil(n_units_total / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()

    for i, uid in enumerate(unit_ids):

        data = results[uid]

        axes[i].plot(data["y_true"], label="True RUL")
        axes[i].plot(data["y_pred"], label="Pred RUL")
        axes[i].set_title(f"Unit {uid}")
        axes[i].legend()

    # remove empty plots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def evaluate_units(model, df, features, config, max_units=None):

    target_scaler = TargetScaler(config["data"]["clip_value"])
    sequence_length = config["data"]["sequence_length"]

    results = {}
    unit_ids = df["unit"].unique()

    # optionally limit number of units
    if max_units is not None:
        unit_ids = unit_ids[:max_units]

    for unit_id in unit_ids:
        unit_df = df[df["unit"] == unit_id]

        # skip short sequences
        if len(unit_df) <= sequence_length:
            continue

        X_unit, y_unit = create_sequences(unit_df, sequence_length, features)

        if len(X_unit) == 0:
            continue

        y_pred = model.predict(X_unit, verbose=0)
        y_pred = target_scaler.inverse_transform(y_pred).flatten()

        mae_cycles = np.mean(np.abs(y_unit - y_pred))
        
        results[unit_id] = {
            "y_pred": y_pred,
            "y_true": y_unit,
            "mae": mae_cycles
        }
    
    print(f"Mean per-unit MAE: {np.mean([r['mae'] for r in results.values()]):.2f}")

    return results