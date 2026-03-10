# Pipeline scripts

Run these from the **project root** in order:

1. **0_merge_data.py** — Merge BTS + YYZ flight data → `data/processed/merged_flights.csv`
2. **1_predict_delays.py** — Train XGBoost, output `data/processed/delays.csv` and `data/figures/feature_importance.png`
3. **2_compute_beta.py** — Compute delay beta scores → `data/processed/beta_scores.csv` and `data/figures/beta_leaderboard.png`
4. **3_optimize_gates.py** — Gate assignment (MILP) → `data/processed/gate_assignments.csv` and `data/figures/gate_optimization_result.png`
5. **embed_data.py** — Build `data.js` from CSVs for the web app

Example:

```bash
python scripts/0_merge_data.py
python scripts/1_predict_delays.py
python scripts/2_compute_beta.py
python scripts/3_optimize_gates.py
python scripts/embed_data.py
```
