"""
YYZ Delay Contagion Optimizer — Streamlit Dashboard.
Visualizes outputs of the gate optimization pipeline (Scripts 1–3).
# pip install streamlit pandas numpy matplotlib pyvis
# Run with: streamlit run dashboard.py (from project root)
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyvis.network import Network
import streamlit.components.v1 as components

BASE = Path(__file__).resolve().parent
DATA_PROCESSED = BASE / "data" / "processed"
DATA_FIGURES = BASE / "data" / "figures"

st.set_page_config(
    page_title="YYZ Delay Contagion Optimizer",
    layout="wide"
)


# ---------------------------------------------------------------------------
# LOAD DATA (with caching)
# ---------------------------------------------------------------------------

@st.cache_data
def load_delays():
    try:
        df = pd.read_csv(str(DATA_PROCESSED / "delays.csv"), low_memory=False)
        if "source" in df.columns:
            df = df[df["source"] == "YYZ"].copy()
        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        raise e


@st.cache_data
def load_beta_scores():
    try:
        df = pd.read_csv(str(DATA_PROCESSED / "beta_scores.csv"), low_memory=False)
        if "source" in df.columns:
            df = df[df["source"] == "YYZ"].copy()
        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        raise e


@st.cache_data
def load_gate_assignments():
    try:
        df = pd.read_csv(str(DATA_PROCESSED / "gate_assignments.csv"), low_memory=False)
        if "source" in df.columns:
            df = df[df["source"] == "YYZ"].copy()
        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        raise e


def main():
    # Load with error handling
    try:
        delays_df = load_delays()
        if delays_df is None:
            st.error("Run scripts 1–3 first to generate delays.csv")
            st.stop()
            return
    except Exception as e:
        st.error(f"Failed to load delays.csv: {e}")
        st.stop()
        return

    try:
        beta_df = load_beta_scores()
        if beta_df is None:
            st.error("Run scripts 1–3 first to generate beta_scores.csv")
            st.stop()
            return
    except Exception as e:
        st.error(f"Failed to load beta_scores.csv: {e}")
        st.stop()
        return

    try:
        gate_assignments_df = load_gate_assignments()
        if gate_assignments_df is None:
            st.error("Run scripts 1–3 first to generate gate_assignments.csv")
            st.stop()
            return
    except Exception as e:
        st.error(f"Failed to load gate_assignments.csv: {e}")
        st.stop()
        return

    # Optional: filter delays_df to YYZ for metrics (already done in load_delays if source exists)
    delays_yyz = delays_df

    # ---------------------------------------------------------------------------
    # HEADER
    # ---------------------------------------------------------------------------
    st.title("YYZ Delay Contagion Optimizer")
    st.markdown(
        "**I4 Case Competition** — Predictive Gate Optimization via Network Contagion Modeling"
    )

    avg_delay = delays_yyz["PREDICTED_DELAY"].mean()
    if pd.isna(avg_delay):
        avg_delay = 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Flights Analyzed", f"{len(delays_df):,}")
    with c2:
        st.metric("Avg Predicted Delay", f"{round(avg_delay, 1)} min")
    with c3:
        st.metric("System R₀ (delay reproduction number)", "0.50")
    with c4:
        st.metric("Gate Conflicts Eliminated (after optimization)", "24 → 0")

    # Override c3/c4 labels per spec
    # (metrics are already shown; spec asked for "delay reproduction number" and "after optimization" as labels)
    # Re-display with correct labels via custom markdown or leave as-is; spec says "Gate Conflicts Eliminated" with label "after optimization" and "System R₀" with "delay reproduction number". We can add small caption. For simplicity the 4 cards are as above; the spec said "Gate Conflicts Eliminated: hardcode '24 → 0' with label 'after optimization'" so the delta text is the label. Good.

    # ---------------------------------------------------------------------------
    # TABS
    # ---------------------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs([
        "🦠 Contagion Network",
        "📊 Beta Leaderboard",
        "🛫 Gate Assignments"
    ])

    # ---------------------------------------------------------------------------
    # TAB 1 — CONTAGION NETWORK
    # ---------------------------------------------------------------------------
    with tab1:
        beta_yyz = beta_df.copy()
        if "departure.scheduledTime.utc" in beta_yyz.columns:
            beta_yyz["departure.scheduledTime.utc"] = pd.to_datetime(
                beta_yyz["departure.scheduledTime.utc"], utc=True, errors="coerce"
            )
        top30 = beta_yyz.nlargest(30, "DELAY_BETA_NORM").reset_index(drop=True)

        net = Network(height="550px", width="100%", bgcolor="#0e1117", font_color="white")
        net.barnes_hut()

        # Node IDs for edge building
        node_ids = []
        for idx, row in top30.iterrows():
            nid = str(row.get("callSign")) if pd.notna(row.get("callSign")) and str(row.get("callSign")).strip() else str(idx)
            node_ids.append(nid)
            label = str(row.get("airline.iata", "")) if pd.notna(row.get("airline.iata")) else str(idx)
            beta_val = float(row.get("DELAY_BETA_NORM", 0) or 0)
            size = 10 + (beta_val / 10)
            if beta_val > 66:
                color = "#ff4b4b"
            elif beta_val > 33:
                color = "#ffa500"
            else:
                color = "#00cc88"
            pred_delay = float(row.get("PREDICTED_DELAY", 0) or 0)
            title = (
                f"Flight: {row.get('callSign', 'N/A')}\n"
                f"Airline: {row.get('airline.iata', 'N/A')}\n"
                f"Predicted Delay: {pred_delay:.0f} min\n"
                f"Beta Score: {beta_val:.1f}"
            )
            net.add_node(nid, label=label, size=size, color=color, title=title)

        # Edges: same aircraft.reg, consecutive in time
        if "aircraft.reg" in top30.columns and "departure.scheduledTime.utc" in top30.columns:
            # Keep index so we can map back to node_ids (built from top30 order)
            top30_sorted = top30.sort_values("departure.scheduledTime.utc")
            for reg, grp in top30_sorted.groupby("aircraft.reg"):
                grp = grp.dropna(subset=["aircraft.reg"])
                if len(grp) < 2:
                    continue
                grp = grp.sort_values("departure.scheduledTime.utc")
                for i in range(len(grp) - 1):
                    idx_a = grp.index[i]
                    idx_b = grp.index[i + 1]
                    id_a = node_ids[idx_a]
                    id_b = node_ids[idx_b]
                    net.add_edge(id_a, id_b, color="#ffffff33")

        DATA_FIGURES.mkdir(parents=True, exist_ok=True)
        contagion_path = DATA_FIGURES / "contagion_network.html"
        net.save_graph(str(contagion_path))
        with open(contagion_path, "r") as f:
            html = f.read()
        components.html(html, height=570)

        col_r0_before, col_r0_after = st.columns(2)
        with col_r0_before:
            st.metric("System R₀ (Before Optimization)", "1.77")
        with col_r0_after:
            st.metric("System R₀ (After Optimization)", "0.50", delta="-1.27", delta_color="inverse")

        st.caption(
            "R₀ is the average number of downstream flights affected when one flight is delayed. "
            "R₀ > 1 means cascading failure. Our optimizer brings R₀ below 1, meaning delay chains "
            "die out rather than spread."
        )

    # ---------------------------------------------------------------------------
    # TAB 2 — BETA LEADERBOARD
    # ---------------------------------------------------------------------------
    with tab2:
        st.subheader("Flight Delay Beta — Systemic Risk Rankings")
        st.markdown(
            "Just as a high-beta stock amplifies market moves, a high-beta flight amplifies "
            "system-wide delays. These are YYZ's most systemically risky flights."
        )

        beta_yyz_t2 = beta_df.copy()
        top15 = beta_yyz_t2.nlargest(15, "DELAY_BETA_NORM")

        display_cols = [
            "callSign", "airline.iata", "airline.name",
            "PREDICTED_DELAY", "downstream_count", "DELAY_BETA_NORM"
        ]
        existing = [c for c in display_cols if c in top15.columns]
        table_df = top15[existing].copy()
        table_df["PREDICTED_DELAY"] = table_df["PREDICTED_DELAY"].round(1)
        table_df["DELAY_BETA_NORM"] = table_df["DELAY_BETA_NORM"].round(1)
        table_df = table_df.rename(columns={
            "callSign": "Flight",
            "airline.iata": "Airline",
            "airline.name": "Carrier",
            "PREDICTED_DELAY": "Predicted Delay (min)",
            "downstream_count": "Downstream Flights at Risk",
            "DELAY_BETA_NORM": "Beta Score (0-100)"
        })
        st.dataframe(table_df, use_container_width=True)

        try:
            st.image(
                str(DATA_FIGURES / "beta_leaderboard.png"),
                caption="Top 15 highest-beta flights at YYZ",
                use_column_width=True
            )
        except Exception:
            st.warning("beta_leaderboard.png not found. Run Script 2 to generate it.")

    # ---------------------------------------------------------------------------
    # TAB 3 — GATE ASSIGNMENTS
    # ---------------------------------------------------------------------------
    with tab3:
        st.subheader("Optimized Gate Assignments")
        col_before, col_after = st.columns(2)
        with col_before:
            st.metric("Conflicts Before", "24")
        with col_after:
            st.metric("Conflicts After", "0", delta="-24", delta_color="inverse")

        try:
            st.image(str(DATA_FIGURES / "gate_optimization_result.png"), use_column_width=True)
        except Exception:
            st.warning("gate_optimization_result.png not found. Run Script 3 to generate it.")

        gate_cols = ["callSign", "airline.iata", "PREDICTED_DELAY", "DELAY_BETA_NORM", "arrival.gate", "ASSIGNED_GATE"]
        gate_existing = [c for c in gate_cols if c in gate_assignments_df.columns]
        st.dataframe(gate_assignments_df[gate_existing], use_container_width=True)

        st.info(
            "Gates are assigned using Mixed-Integer Linear Programming (MILP). High-beta flights "
            "receive buffered time windows proportional to their predicted delay, preventing "
            "cascade conflicts before they occur."
        )


if __name__ == "__main__":
    main()

# Run with: streamlit run dashboard.py
