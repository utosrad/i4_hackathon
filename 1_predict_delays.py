"""
YYZ Flight Arrival Delay Prediction.
Uses merged flight data (BTS + YYZ); trains XGBoost and outputs delays.csv + feature_importance.png.
# pip install pandas numpy xgboost scikit-learn matplotlib
"""

import sys
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# User updates this path to their CSV location (run 0_merge_data.py first to create merged_flights.csv)
FILE_PATH = "merged_flights.csv"

REQUIRED_COLUMNS = [
    "aircraft.reg",
    "airline.iata",
    "departure.airport.iata",
    "arrival.scheduledTime.utc",
    "arrival.revisedTime.utc",
    "departure.scheduledTime.utc",
    "ARR_DELAY",
]

UTC_COLUMNS = [
    "departure.scheduledTime.utc",
    "departure.revisedTime.utc",
    "arrival.scheduledTime.utc",
    "arrival.revisedTime.utc",
]

FEATURES = [
    "PRIOR_LEG_DELAY",
    "DEP_HOUR",
    "IS_PEAK_HOUR",
    "CARRIER_ENCODED",
    "ORIGIN_ENCODED",
    "IS_WIDEBODY",
    "DEP_RUNWAY_DELAY",
]
TARGET = "ARR_DELAY"

WIDEBODY_MODELS = ["777", "787", "747", "A330", "A340", "A350", "A380"]


def main():
    # --- Step 1: Load & validate ---
    print("Loading data...")
    df = pd.read_csv(FILE_PATH, low_memory=False)
    print(df.columns.tolist())
    print(df.shape)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        print("Missing required columns:", sorted(missing))
        sys.exit(1)

    # Parse UTC datetime columns (format='mixed' handles BTS vs YYZ datetime formats)
    for col in UTC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="mixed", utc=True, errors="coerce")
    if "departure.runwayTime.utc" in df.columns:
        df["departure.runwayTime.utc"] = pd.to_datetime(
            df["departure.runwayTime.utc"], format="mixed", utc=True, errors="coerce"
        )

    # Drops (do not require aircraft.reg — BTS rows have no tail number)
    if "status" in df.columns:
        df = df[~df["status"].fillna("").str.lower().str.contains("cancel")]
    else:
        pass  # no status column, skip cancellation filter

    df = df.sort_values(["aircraft.reg", "departure.scheduledTime.utc"]).reset_index(
        drop=True
    )

    # --- Step 2: ARR_DELAY (use as-is from merged_flights.csv; no recomputation) ---
    df = df.dropna(subset=["ARR_DELAY"])
    df = df[df["ARR_DELAY"] <= 600]
    print(f"Cleaning data... {len(df)} rows remaining after drops")
    print("ARR_DELAY stats:", df["ARR_DELAY"].describe())

    # --- Step 3: Feature engineering ---
    print("Engineering features...")

    df["PRIOR_LEG_DELAY"] = df.groupby("aircraft.reg")["ARR_DELAY"].shift(1)
    df["PRIOR_LEG_DELAY"] = df["PRIOR_LEG_DELAY"].fillna(0)

    df["DEP_HOUR"] = df["departure.scheduledTime.utc"].dt.hour
    df["IS_PEAK_HOUR"] = (
        ((df["DEP_HOUR"] >= 7) & (df["DEP_HOUR"] <= 9))
        | ((df["DEP_HOUR"] >= 16) & (df["DEP_HOUR"] <= 19))
    ).astype(int)

    le_carrier = LabelEncoder()
    df["CARRIER_ENCODED"] = le_carrier.fit_transform(
        df["airline.iata"].fillna("").astype(str)
    )
    le_origin = LabelEncoder()
    df["ORIGIN_ENCODED"] = le_origin.fit_transform(
        df["departure.airport.iata"].fillna("").astype(str)
    )

    model_str = df["aircraft.model"].fillna("").astype(str)
    df["IS_WIDEBODY"] = 0
    for wb in WIDEBODY_MODELS:
        df.loc[model_str.str.contains(wb, regex=False), "IS_WIDEBODY"] = 1

    if "departure.runwayTime.utc" in df.columns:
        df["DEP_RUNWAY_DELAY"] = (
            df["departure.runwayTime.utc"] - df["departure.scheduledTime.utc"]
        ).dt.total_seconds() / 60.0
        df["DEP_RUNWAY_DELAY"] = df["DEP_RUNWAY_DELAY"].fillna(
            df["DEP_RUNWAY_DELAY"].median()
        )
    else:
        df["DEP_RUNWAY_DELAY"] = 0

    # --- Step 4: Train/test split (time series) ---
    df = df.sort_values("departure.scheduledTime.utc").reset_index(drop=True)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    # --- Step 5: XGBoost ---
    print("Training XGBoost model...")
    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    model.fit(train_df[FEATURES], train_df[TARGET])
    pred_test = model.predict(test_df[FEATURES])
    mae = mean_absolute_error(test_df[TARGET], pred_test)
    rmse = np.sqrt(mean_squared_error(test_df[TARGET], pred_test))
    print(f"Test MAE: {mae:.1f} min | Test RMSE: {rmse:.1f} min")

    # Feature importance plot
    importance = model.feature_importances_
    plt.figure(figsize=(8, 5))
    plt.barh(FEATURES, importance)
    plt.xlabel("Feature importance")
    plt.title("XGBoost feature importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()

    # --- Step 6: Output delays.csv ---
    df["PREDICTED_DELAY"] = np.maximum(0, model.predict(df[FEATURES]))
    out_cols = [
        "callSign",
        "aircraft.reg",
        "airline.iata",
        "airline.name",
        "departure.airport.iata",
        "arrival.airport.iata",
        "departure.scheduledTime.utc",
        "arrival.scheduledTime.utc",
        "departure.gate",
        "arrival.gate",
        "ARR_DELAY",
        "PRIOR_LEG_DELAY",
        "PREDICTED_DELAY",
        "source",
    ]
    df_out = df.reindex(columns=out_cols)
    print("Saving delays.csv...")
    df_out.to_csv("delays.csv", index=False)
    print(f"delays.csv saved with {len(df_out)} rows")
    print("Done.")


if __name__ == "__main__":
    main()
