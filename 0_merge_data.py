"""
Merge BTS flight delay dataset with YYZ flight movements into a single enriched training file.
"""

import sys
import pandas as pd

# ---------------------------------------------------------------------------
# FILE PATH CONFIG (user updates these)
# ---------------------------------------------------------------------------
BTS_FILE = "flights_sample_3m.csv"   # large Kaggle delay dataset
YYZ_FILE = "yyz_flights.csv"         # original YYZ dataset
OUTPUT_FILE = "merged_flights.csv"

# Common schema columns for concatenation (same order in both dataframes)
COMMON_COLUMNS = [
    "departure.scheduledTime.utc",
    "aircraft.reg",
    "airline.iata",
    "airline.name",
    "callSign",
    "departure.airport.iata",
    "arrival.airport.iata",
    "arrival.scheduledTime.utc",
    "arrival.revisedTime.utc",
    "departure.gate",
    "arrival.gate",
    "aircraft.model",
    "ARR_DELAY",
    "DEP_DELAY",
    "DISTANCE",
    "source",
]

REQUIRED_BTS_COLUMNS = [
    "FL_DATE",
    "TAIL_NUM",
    "OP_CARRIER",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "DEP_DELAY",
    "ARR_DELAY",
    "CANCELLED",
    "DISTANCE",
    "DEP_TIME",
    "ARR_TIME",
]


def main():
    # -----------------------------------------------------------------------
    # STEP 1 — LOAD BTS DATASET
    # -----------------------------------------------------------------------
    print("Loading BTS dataset (this may take a moment)...")
    df_bts = pd.read_csv(BTS_FILE)
    print(f"BTS loaded: {len(df_bts)} rows")

    # Optional compatibility: repo's file uses AIRLINE_CODE and may lack TAIL_NUM
    if "OP_CARRIER" not in df_bts.columns and "AIRLINE_CODE" in df_bts.columns:
        df_bts = df_bts.rename(columns={"AIRLINE_CODE": "OP_CARRIER"})
    if "TAIL_NUM" not in df_bts.columns:
        df_bts["TAIL_NUM"] = ""

    missing = [c for c in REQUIRED_BTS_COLUMNS if c not in df_bts.columns]
    if missing:
        print(f"Missing required BTS columns: {missing}")
        sys.exit(1)

    df_bts = df_bts[df_bts["CANCELLED"] != 1]
    df_bts = df_bts.dropna(subset=["ARR_DELAY", "TAIL_NUM"])
    df_bts["FL_DATE"] = pd.to_datetime(df_bts["FL_DATE"])
    print(f"BTS rows after drops: {len(df_bts)}")

    # -----------------------------------------------------------------------
    # STEP 2 — STANDARDIZE BTS TO COMMON SCHEMA
    # -----------------------------------------------------------------------
    print("Standardizing BTS schema...")
    df_bts = df_bts.rename(
        columns={
            "FL_DATE": "departure.scheduledTime.utc",
            "TAIL_NUM": "aircraft.reg",
            "OP_CARRIER": "airline.iata",
            "ORIGIN": "departure.airport.iata",
            "DEST": "arrival.airport.iata",
            "ARR_DELAY": "ARR_DELAY",
            "DEP_DELAY": "DEP_DELAY",
            "DISTANCE": "DISTANCE",
        }
    )
    df_bts["arrival.gate"] = None
    df_bts["departure.gate"] = None
    df_bts["airline.name"] = None
    df_bts["callSign"] = None
    df_bts["aircraft.model"] = None
    df_bts["arrival.scheduledTime.utc"] = None
    df_bts["arrival.revisedTime.utc"] = None
    df_bts["source"] = "BTS"

    df_bts = df_bts[COMMON_COLUMNS]

    print("BTS ARR_DELAY non-null count:", df_bts["ARR_DELAY"].notna().sum())
    print("BTS ARR_DELAY sample:", df_bts["ARR_DELAY"].head(10).tolist())

    # -----------------------------------------------------------------------
    # STEP 3 — LOAD & STANDARDIZE YYZ DATASET
    # -----------------------------------------------------------------------
    print("Loading YYZ dataset...")
    df_yyz = pd.read_csv(YYZ_FILE)

    print("Computing YYZ arrival delays...")
    df_yyz["arrival.scheduledTime.utc"] = pd.to_datetime(
        df_yyz["arrival.scheduledTime.utc"], utc=True
    )
    df_yyz["arrival.revisedTime.utc"] = pd.to_datetime(
        df_yyz["arrival.revisedTime.utc"], utc=True
    )
    df_yyz["ARR_DELAY"] = (
        df_yyz["arrival.revisedTime.utc"] - df_yyz["arrival.scheduledTime.utc"]
    ).dt.total_seconds() / 60

    df_yyz = df_yyz.dropna(subset=["ARR_DELAY"])
    df_yyz = df_yyz[(df_yyz["ARR_DELAY"] >= -30) & (df_yyz["ARR_DELAY"] <= 600)]

    df_yyz["DEP_DELAY"] = None
    df_yyz["DISTANCE"] = None
    df_yyz["source"] = "YYZ"

    df_yyz = df_yyz[COMMON_COLUMNS]
    print(f"YYZ rows: {len(df_yyz)}")

    # -----------------------------------------------------------------------
    # STEP 4 — CONCATENATE
    # -----------------------------------------------------------------------
    print("Merging datasets...")
    if len(df_bts) == 0:
        merged = df_yyz.copy()
    elif len(df_yyz) == 0:
        merged = df_bts.copy()
    else:
        merged = pd.concat([df_bts, df_yyz], ignore_index=True)
    merged = merged.sort_values(
        by=["aircraft.reg", "departure.scheduledTime.utc"],
        na_position="last",
    )

    print(f"Final merged shape: {len(merged)} rows, {len(merged.columns)} columns")
    print(merged["source"].value_counts())
    print(merged["ARR_DELAY"].describe())

    # -----------------------------------------------------------------------
    # STEP 5 — SAVE
    # -----------------------------------------------------------------------
    print("Saving merged_flights.csv...")
    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"merged_flights.csv saved with {len(merged)} rows")

    merged = pd.read_csv(OUTPUT_FILE, nrows=5)
    print("Merged ARR_DELAY sample:", merged["ARR_DELAY"].tolist())
    print(
        "Merged ARR_DELAY non-null count:",
        pd.read_csv(OUTPUT_FILE)["ARR_DELAY"].notna().sum(),
    )
    print("Done.")


if __name__ == "__main__":
    main()
