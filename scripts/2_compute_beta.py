"""
Compute Delay Beta scores from delays.csv (output of Script 1).
Outputs beta_scores.csv and beta_leaderboard.png.
Run from project root: python scripts/2_compute_beta.py
# pip install pandas numpy matplotlib
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE / "data" / "processed"
DATA_FIGURES = BASE / "data" / "figures"


def main():
    # --- Step 1: Load ---
    print("Loading delays.csv...")
    df = pd.read_csv(str(DATA_PROCESSED / "delays.csv"))
    df["departure.scheduledTime.utc"] = pd.to_datetime(
        df["departure.scheduledTime.utc"], utc=True
    )
    df["arrival.scheduledTime.utc"] = pd.to_datetime(
        df["arrival.scheduledTime.utc"], utc=True
    )
    df = df.sort_values(
        ["aircraft.reg", "departure.scheduledTime.utc"]
    ).reset_index(drop=True)
    print("Shape:", df.shape)

    if len(df) == 0:
        print("No rows in delays.csv. Exiting.")
        return

    # --- Step 2: Compute downstream count ---
    df["downstream_count"] = df.groupby("aircraft.reg").cumcount(ascending=False)

    # --- Step 3: Compute Beta and normalized Beta ---
    df["DELAY_BETA"] = df["PREDICTED_DELAY"] * (1 + df["downstream_count"])
    span = df["DELAY_BETA"].max() - df["DELAY_BETA"].min()
    if span == 0:
        df["DELAY_BETA_NORM"] = 0
    else:
        df["DELAY_BETA_NORM"] = (
            100
            * (df["DELAY_BETA"] - df["DELAY_BETA"].min())
            / span
        )

    # --- Step 4: R0 ---
    delayed = df[df["PREDICTED_DELAY"] > 15]
    R0 = delayed["downstream_count"].mean() if len(delayed) > 0 else 0.0
    print(f"System R0: {R0:.2f}")
    print(
        f"Interpretation: each delayed flight affects {R0:.2f} "
        "downstream flights on average"
    )

    # --- Step 5: Output CSV and top 10 ---
    df.to_csv(str(DATA_PROCESSED / "beta_scores.csv"), index=False)
    print(f"beta_scores.csv saved with {len(df)} rows")

    top10 = df.nlargest(10, "DELAY_BETA_NORM")[
        ["callSign", "airline.iata", "PREDICTED_DELAY", "downstream_count", "DELAY_BETA_NORM"]
    ]
    print("\nTop 10 highest beta flights:")
    print(top10.to_string(index=False))

    # --- Step 6: Plot ---
    top15 = df.nlargest(15, "DELAY_BETA_NORM")
    airlines = top15["airline.iata"].astype(str).unique()
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(airlines), 1)))
    airline_to_color = dict(zip(airlines, colors[: len(airlines)]))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        range(len(top15)),
        top15["DELAY_BETA_NORM"].values,
        color=[airline_to_color.get(str(a), "gray") for a in top15["airline.iata"]],
    )
    ax.set_xticks(range(len(top15)))
    ax.set_xticklabels(top15["callSign"].fillna("").astype(str), rotation=45, ha="right")
    ax.set_ylabel("DELAY_BETA_NORM")
    ax.set_title("Delay Beta Leaderboard — Top 15 High-Risk Flights")

    # Legend by airline
    legend_elements = [
        Patch(facecolor=airline_to_color[a], label=a) for a in airlines
    ]
    ax.legend(handles=legend_elements, title="airline.iata")

    plt.tight_layout()
    DATA_FIGURES.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(DATA_FIGURES / "beta_leaderboard.png"), dpi=150)
    plt.close()
    print("\nbeta_leaderboard.png saved.")


if __name__ == "__main__":
    main()
