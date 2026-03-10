"""
Merge BTS flight delay dataset with YYZ flight movements into a single enriched training file.
Run from project root: python scripts/0_merge_data.py
"""

import os
import sys
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA_RAW = BASE / "data" / "raw"
DATA_PROCESSED = BASE / "data" / "processed"

BTS_FILE = DATA_RAW / "flights_sample_3m.csv"
YYZ_FILE = DATA_RAW / "yyz_flights.csv"
OUTPUT_FILE = DATA_PROCESSED / "merged_flights.csv"

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
    # CHECK REQUIRED FILES EXIST
    # -----------------------------------------------------------------------
    if not BTS_FILE.is_file():
        print(f"Error: Missing '{BTS_FILE}'")
        print("  Script 0 needs a BTS/Kaggle flight delay CSV (e.g. from Kaggle 'Flight Delays' or similar).")
        print("  Place the file in this directory or set BTS_FILE at the top of this script.")
        sys.exit(1)
    if not YYZ_FILE.is_file():
        print(f"Error: Missing '{YYZ_FILE}'")
        print("  Place the YYZ flight movements CSV in this directory.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # STEP 1 — LOAD BTS DATASET
    # -----------------------------------------------------------------------
    print("Loading BTS dataset (this may take a moment)...")
    df_bts = pd.read_csv(str(BTS_FILE))
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
    df_yyz = pd.read_csv(str(YYZ_FILE))

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
    merged.to_csv(str(OUTPUT_FILE), index=False)
    print(f"merged_flights.csv saved with {len(merged)} rows")

    merged = pd.read_csv(str(OUTPUT_FILE), nrows=5)
    print("Merged ARR_DELAY sample:", merged["ARR_DELAY"].tolist())
    print(
        "Merged ARR_DELAY non-null count:",
        pd.read_csv(str(OUTPUT_FILE))["ARR_DELAY"].notna().sum(),
    )
    print("Done.")


if __name__ == "__main__":
    main()
