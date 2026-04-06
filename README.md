# AI Maritime Congestion & Delay Prediction System

## Project Overview

This project builds an end-to-end AI-driven system that predicts port congestion and vessel arrival delays using historical AIS (Automatic Identification System) data, weather signals, and trade flow patterns. It also generates optimized vessel scheduling recommendations and explains predictions using interpretable machine learning techniques.

The system helps port planners and logistics managers make proactive decisions — reducing waiting times, minimizing costs, and improving supply chain reliability.

---

## Files Included

- `phase1.py` → Data loading, cleaning, EDA, and voyage aggregation
- `phase2.py` → Feature engineering and enrichment
- `phase2_clustering.py` → K-Means vessel clustering and NetworkX port flow graph
- `phase2_anomaly.py` → Isolation Forest and Z-score anomaly detection
- `phase3.py` → Machine learning model training (RandomForest with CV and threshold tuning)
- `phase3_optimizer.py` → Constraint-based greedy schedule optimizer
- `phase4_forecast.py` → Time-series congestion forecasting (Seasonal EWM / Prophet)
- `dashboard.py` → Interactive 8-page Streamlit dashboard
- `sample_1000.csv` → Raw AIS dataset used for training and evaluation
- `requirements.txt` → Required Python libraries

---

## Technologies Used

- Python
- Pandas & NumPy
- Scikit-learn
- NetworkX
- Streamlit
- Matplotlib / Seaborn
- Prophet *(optional — fallback baseline included)*
- Joblib

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/maritime-ai-delay-prediction.git
   cd maritime-ai-delay-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run each phase in order:
   ```bash
   python phase1.py
   python phase2.py
   python phase2_clustering.py
   python phase2_anomaly.py
   python phase3.py
   python phase3_optimizer.py
   python phase4_forecast.py
   ```

4. Launch the dashboard:
   ```bash
   streamlit run dashboard.py
   ```

---

## Dataset

The dataset contains raw AIS (Automatic Identification System) ping records from the **Helsinki ↔ Tallinn** ferry corridor (2018–2019), covering 4 vessels across approximately 900 voyages.

| Column | Description |
|--------|-------------|
| `vessel_id / ship` | Unique vessel identifier |
| `origin_port` | Departure port code |
| `destination_port` | Arrival port code |
| `scheduled_arrival` | Planned arrival datetime |
| `actual_arrival` | Real arrival datetime |
| `delay_minutes` | Target variable — departure delay in minutes |
| `weather_disruption_index` | Proxy score for adverse weather conditions |
| `trade_flow_demand_teu` | Port congestion demand indicator |
| `port_capacity_teu` | Port handling capacity |
| `turnaround_time_hours` | Vessel loading/unloading duration |
| `vessel_priority_level` | Scheduling priority (higher = prioritized docking) |

---

## Features

- Voyage-level delay prediction using 21 engineered features
- Vessel behaviour clustering (K-Means with silhouette scoring)
- Port bottleneck detection using NetworkX graph centrality
- Anomaly detection via Isolation Forest and Z-score analysis
- Weather-proxy delay spike analysis
- Constraint-based schedule optimiser with vessel priority and port capacity limits
- 30-day ahead congestion forecasting
- Interactive Streamlit dashboard with 8 analytical pages

---

## Output

- Predicts whether a vessel departure will be delayed (classification)
- Estimates delay magnitude in minutes (regression-ready)
- Recommends optimal departure windows per vessel
- Generates rerouting suggestions to low-congestion time slots
- Displays port congestion heatmaps, anomaly records, and forecast curves

---

## Model Performance

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.897 |
| Recall | 86.5% |
| F1 Score | 0.542 |
| Train-Test Gap | 9.8pp ✅ |

> Decision threshold tuned to 0.30 to maximise recall — missing a delay is ~10× more costly than a false alarm in port operations.

---

## Key Technical Decisions

- **Date parsing fix**: `ata` column uses `MM/DD/YYYY` (not `DD/MM/YYYY` like other columns) — incorrect parsing caused ±465,000 minute errors in the original code
- **Target variable**: `dep_delay_min > 5 min` used instead of arrival delay (59% of arrival records were missing)
- **No data leakage**: `vessel_avg_delay` recomputed on train split only; all lag features use `shift(1)`
- **Chronological split**: 70/30 time-ordered split — no shuffling on time-series data

---

## Future Improvements

- Real-time AIS data integration via live API
- LSTM / Transformer model for sequence-based delay prediction
- SHAP explainability layer for feature attribution
- Multi-port corridor expansion beyond Helsinki ↔ Tallinn
- Dockerized deployment for production environments

---

## Author

Swayam Handoo
WiseAnalytics Intern

