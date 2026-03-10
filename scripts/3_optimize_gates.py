"""
Assign gates to flights using beta scores from Script 2, minimizing delay cascade risk.
Uses MILP (PuLP) to avoid gate conflicts while considering DELAY_BETA_NORM.
Run from project root: python scripts/3_optimize_gates.py
# pip install pulp pandas numpy matplotlib
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pulp
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE / "data" / "processed"
DATA_FIGURES = BASE / "data" / "figures"


def conflicts(f1, f2):
    """Returns True if flight f1 and f2 have overlapping time windows."""
    start1 = f1["departure.scheduledTime.utc"]
    end1 = f1["buffered_arrival"]
    start2 = f2["departure.scheduledTime.utc"]
    end2 = f2["buffered_arrival"]
    return not (end1 <= start2 or end2 <= start1)


def count_conflicts(df, gate_col):
    """Count conflicts when flights share the same gate and time windows overlap."""
    conflict_count = 0
    rows = df.reset_index(drop=True)
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if rows.loc[i, gate_col] == rows.loc[j, gate_col]:
                if conflicts(rows.iloc[i], rows.iloc[j]):
                    conflict_count += 1
    return conflict_count


def main():
    # --- STEP 1: Load & prep ---
    print("Loading beta_scores.csv...")
    df = pd.read_csv(str(DATA_PROCESSED / "beta_scores.csv"))
    df["departure.scheduledTime.utc"] = pd.to_datetime(
        df["departure.scheduledTime.utc"], utc=True
    )
    df["arrival.scheduledTime.utc"] = pd.to_datetime(
        df["arrival.scheduledTime.utc"], utc=True
    )
    df = df.dropna(subset=["departure.scheduledTime.utc", "arrival.scheduledTime.utc"])
    df["buffered_arrival"] = df["arrival.scheduledTime.utc"] + pd.to_timedelta(
        df["PREDICTED_DELAY"], unit="m"
    )
    print("Shape:", df.shape)

    # --- STEP 2: Define gates ---
    observed_gates = df["arrival.gate"].dropna().unique().tolist()
    gates = observed_gates.copy()
    if len(gates) < 5:
        gates = gates + ["G1", "G2", "G3", "G4", "G5"]
    gates = list(set(gates))
    print(f"Gates available: {len(gates)} gates → {gates}")

    # --- STEP 3: Sample for MILP ---
    flights = df.nlargest(50, "DELAY_BETA_NORM").reset_index(drop=True)
    print("Optimizing gate assignments for 50 highest-beta flights")

    # --- STEP 4: Conflict check is defined above (conflicts) ---

    # --- STEP 5: Build MILP with PuLP ---
    prob = pulp.LpProblem("GateAssignment", pulp.LpMinimize)

    # Decision variables: x[i][g] = 1 if flight i assigned to gate g
    x = {}
    for i in range(len(flights)):
        for g in gates:
            x[i, g] = pulp.LpVariable(f"x_{i}_{g}", cat="Binary")

    # Objective: minimize sum of DELAY_BETA_NORM for each assignment
    prob += pulp.lpSum(
        flights.loc[i, "DELAY_BETA_NORM"] * x[i, g]
        for i in range(len(flights))
        for g in gates
    )

    # Constraint 1: each flight gets exactly one gate
    for i in range(len(flights)):
        prob += pulp.lpSum(x[i, g] for g in gates) == 1

    # Constraint 2: conflicting flights cannot share a gate
    for i in range(len(flights)):
        for j in range(i + 1, len(flights)):
            if conflicts(flights.iloc[i], flights.iloc[j]):
                for g in gates:
                    prob += x[i, g] + x[j, g] <= 1

    # Solve
    print("Running optimizer...")
    solver_ok = False
    try:
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        solver_ok = prob.status == pulp.LpStatusOptimal
    except Exception as e:
        print(f"CBC solver unavailable ({e}), trying default...")
    if not solver_ok:
        try:
            prob.solve()
            solver_ok = prob.status == pulp.LpStatusOptimal
        except Exception as e2:
            print(f"Default solver failed: {e2}")
    if not solver_ok:
        print("No MILP solver available; using greedy gate assignment.")
        # Greedy: sort by departure, assign each flight to first gate with no overlapping window
        assigned_gates = [None] * len(flights)
        gate_windows = {g: [] for g in gates}  # list of (start, end) per gate
        order = flights["departure.scheduledTime.utc"].argsort()
        for idx in order:
            row = flights.iloc[idx]
            start, end = row["departure.scheduledTime.utc"], row["buffered_arrival"]
            chosen = None
            for g in gates:
                if all(end <= s or e <= start for s, e in gate_windows[g]):
                    chosen = g
                    gate_windows[g].append((start, end))
                    break
            assigned_gates[idx] = chosen if chosen else "UNASSIGNED"
    else:
        print(f"Solver status: {pulp.LpStatus[prob.status]}")
        # --- STEP 6: Extract results from MILP ---
        assigned_gates = []
        for i in range(len(flights)):
            for g in gates:
                if pulp.value(x[i, g]) == 1:
                    assigned_gates.append(g)
                    break
            else:
                assigned_gates.append("UNASSIGNED")

    flights = flights.copy()
    flights["ASSIGNED_GATE"] = assigned_gates

    # --- STEP 7: Compute before/after conflicts ---
    flights_with_orig_gate = flights[flights["arrival.gate"].notna()]
    before = count_conflicts(flights_with_orig_gate, "arrival.gate")
    after = count_conflicts(flights, "ASSIGNED_GATE")

    print(f"Gate conflicts BEFORE optimization: {before}")
    print(f"Gate conflicts AFTER  optimization: {after}")
    print(
        f"Conflict reduction: {round((before - after) / max(before, 1) * 100, 1)}%"
    )

    # --- STEP 8: Output ---
    out_cols = [
        "callSign",
        "airline.iata",
        "departure.scheduledTime.utc",
        "arrival.scheduledTime.utc",
        "buffered_arrival",
        "PREDICTED_DELAY",
        "DELAY_BETA_NORM",
        "arrival.gate",
        "ASSIGNED_GATE",
    ]
    flights[out_cols].to_csv(str(DATA_PROCESSED / "gate_assignments.csv"), index=False)
    print("gate_assignments.csv saved")

    # --- STEP 9: Plot ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([0], [before], label="Before", color="tab:orange", width=0.4)
    ax.bar([1], [after], label="After", color="tab:green", width=0.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Before", "After"])
    ax.set_ylabel("Conflict count")
    ax.set_title("Gate Conflicts: Before vs After Optimization")
    ax.legend()
    plt.tight_layout()
    DATA_FIGURES.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(DATA_FIGURES / "gate_optimization_result.png"), dpi=150)
    plt.close()
    print("gate_optimization_result.png saved")


if __name__ == "__main__":
    main()
