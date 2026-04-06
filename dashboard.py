import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, json

st.set_page_config(
    page_title="Port Commander",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root{
  --bg:#0a0e1a;--card:#111827;--card2:#151d2e;
  --border:#1e2d45;--brite:#2a3f5f;
  --cyan:#00d4ff;--cyan2:#0099bb;--amber:#f59e0b;
  --green:#10b981;--red:#ef4444;
  --tp:#e2e8f0;--ts:#94a3b8;--tm:#475569;
  --mono:'Space Mono',monospace;--sans:'DM Sans',sans-serif;
}
html,body,[class*="css"]{font-family:var(--sans);background:var(--bg);color:var(--tp);}
.stApp{background:var(--bg);
  background-image:radial-gradient(ellipse at 20% 0%,rgba(0,212,255,.04) 0%,transparent 60%),
                   radial-gradient(ellipse at 80% 100%,rgba(245,158,11,.03) 0%,transparent 60%);}

/* ── Hide only the Streamlit menu & footer — NOT the header ── */
/* Keeping the header visible preserves the sidebar toggle button */
#MainMenu{visibility:hidden;}
footer{visibility:hidden;}

/* ── Style the Streamlit header bar to match our dark theme ── */
header[data-testid="stHeader"]{
  background:#0a0e1a!important;
  border-bottom:1px solid #1e2d45!important;
}

/* ── Style the sidebar toggle arrow button to match our theme ── */
header[data-testid="stHeader"] button,
header[data-testid="stHeader"] button:focus,
header[data-testid="stHeader"] button:hover{
  background:rgba(0,153,187,0.20)!important;
  border-radius:6px!important;
  border:1px solid #2a3f5f!important;
  color:#00d4ff!important;
}
header[data-testid="stHeader"] button svg{
  fill:#00d4ff!important;
  color:#00d4ff!important;
}
header[data-testid="stHeader"] button:hover{
  background:rgba(0,212,255,0.15)!important;
  border-color:#00d4ff!important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"]{
  background:var(--card)!important;
  border-right:1px solid var(--border)!important;
}
section[data-testid="stSidebar"] *{color:var(--tp)!important;}

div[role="radiogroup"] label{background:transparent!important;border:1px solid transparent!important;
  border-radius:6px!important;padding:8px 12px!important;margin-bottom:4px!important;
  font-size:.88rem!important;color:var(--ts)!important;transition:all .15s!important;}
div[role="radiogroup"] label:hover{background:rgba(0,212,255,.06)!important;border-color:var(--brite)!important;color:var(--tp)!important;}
div[role="radiogroup"] label[data-checked="true"]{background:rgba(0,212,255,.10)!important;border-color:var(--cyan2)!important;color:var(--cyan)!important;}
.block-container{padding-top:1.8rem!important;padding-bottom:2rem!important;max-width:1280px!important;}
[data-testid="metric-container"]{background:var(--card)!important;border:1px solid var(--border)!important;
  border-radius:10px!important;padding:1.1rem 1.3rem!important;}
[data-testid="metric-container"] label{font-size:.72rem!important;font-weight:600!important;
  letter-spacing:.08em!important;text-transform:uppercase!important;color:var(--tm)!important;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{font-family:var(--mono)!important;font-size:1.6rem!important;}
.stButton>button{background:linear-gradient(135deg,var(--cyan2),#006688)!important;color:#fff!important;
  border:none!important;border-radius:6px!important;font-weight:600!important;
  padding:.55rem 1.4rem!important;transition:all .2s!important;}
.stButton>button:hover{transform:translateY(-1px)!important;box-shadow:0 4px 20px rgba(0,212,255,.25)!important;}
[data-testid="stDataFrame"]{border:1px solid var(--border)!important;border-radius:8px!important;overflow:hidden!important;}
[data-testid="stDataFrame"] th{background:var(--card2)!important;color:var(--tm)!important;
  font-size:.72rem!important;font-weight:600!important;letter-spacing:.08em!important;text-transform:uppercase!important;}
[data-testid="stDataFrame"] td{color:var(--tp)!important;font-family:var(--mono)!important;font-size:.82rem!important;}
[data-testid="stExpander"]{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:8px!important;}
hr{border-color:var(--border)!important;margin:1.4rem 0!important;}
label[data-testid="stWidgetLabel"] p{font-size:.78rem!important;font-weight:600!important;color:var(--ts)!important;}
.pc-title{font-family:var(--mono);font-size:1.5rem;font-weight:700;margin-bottom:.25rem;}
.pc-sub{font-size:.85rem;color:var(--tm);margin-bottom:1.5rem;}
.pc-sec{font-size:.7rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
  color:var(--tm);margin-bottom:.6rem;display:flex;align-items:center;gap:6px;}
.pc-sec::after{content:'';flex:1;height:1px;background:var(--border);}
.pc-card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:1.2rem 1.4rem;margin-bottom:1rem;}
.pc-card-cy{border-left:3px solid var(--cyan);}
.pc-card-am{border-left:3px solid var(--amber);}
.pc-card-rd{border-left:3px solid var(--red);}
.pc-card-gr{border-left:3px solid var(--green);}
.pc-badge{display:inline-block;font-family:var(--mono);font-size:.68rem;font-weight:700;
  letter-spacing:.06em;padding:2px 8px;border-radius:3px;text-transform:uppercase;}
.cy{background:rgba(0,212,255,.12);color:var(--cyan);border:1px solid rgba(0,212,255,.3);}
.am{background:rgba(245,158,11,.12);color:var(--amber);border:1px solid rgba(245,158,11,.3);}
.gr{background:rgba(16,185,129,.12);color:var(--green);border:1px solid rgba(16,185,129,.3);}
.rd{background:rgba(239,68,68,.12);color:var(--red);border:1px solid rgba(239,68,68,.3);}
.pc-rb{background:var(--card2);border:1px solid var(--border);border-radius:10px;padding:1.4rem 1.6rem;text-align:center;}
.pc-rl{font-size:.7rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--tm);margin-bottom:.4rem;}
.pc-rv{font-family:var(--mono);font-size:2rem;font-weight:700;}
.pc-rs{font-size:.8rem;color:var(--tm);margin-top:.3rem;}
.pc-ib{background:var(--card2);border:1px solid var(--border);border-radius:8px;padding:1rem 1.2rem;margin-bottom:.5rem;}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────
# Typical vessel delays computed from actual data (used as fallback)
VESSEL_TYPICAL = {"Megastar": -9.5, "Star": -8.8, "Finlandia": -7.5, "Europa": -6.0}
DAY_MAP = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

MODEL_FEATURES = [
    "mean_sog","is_slow_crossing",
    "hour_of_dep","day_of_week","month","is_weekend","is_peak_hour",
    "hour_sin","hour_cos","dow_sin","dow_cos",
    "port_traffic","traffic_3h","traffic_6h","prev_traffic",
    "vessel_avg_delay","prev_dep_delay","rolling_delay_3v","prev_was_late",
    "is_hel_to_tal","sched_duration_min",
]

# ── Loaders ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    p = "outputs/voyage_dataset.csv"
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, parse_dates=["etd_sched"])

@st.cache_resource
def load_model_bundle():
    p = "outputs/congestion_model.pkl"
    if not os.path.exists(p):
        return None
    return joblib.load(p)

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_forecast():
    p = "outputs/congestion_forecast.csv"
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, parse_dates=["date"])

@st.cache_data
def load_schedule():
    p = "outputs/optimized_schedule.csv"
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, parse_dates=["etd_sched", "optimized_etd"])

@st.cache_data
def load_clusters():
    p = "outputs/vessel_clusters.csv"
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)

@st.cache_data
def compute_traffic_defaults(_df):
    d = {}
    for c in ("port_traffic","traffic_3h","traffic_6h","prev_traffic"):
        d[c] = float(_df[c].mean()) if (_df is not None and c in _df.columns) else 3.0
    return d

@st.cache_data
def compute_vessel_avg_delay(_df):
    if _df is not None and "dep_delay_min" in _df.columns and "ship" in _df.columns:
        return _df.groupby("ship")["dep_delay_min"].mean().to_dict()
    return VESSEL_TYPICAL

def build_feature_row(vessel, hour, day_of_week, month, mean_sog, td, vd,
                      prev_dep_delay=0.0, rolling_delay=0.0, prev_was_late=0,
                      is_hel_to_tal=1, sched_duration=120.0):
    row = {
        "mean_sog":          mean_sog,
        "is_slow_crossing":  1 if mean_sog < 15 else 0,
        "hour_of_dep":       hour,
        "day_of_week":       day_of_week,
        "month":             month,
        "is_weekend":        1 if day_of_week >= 5 else 0,
        "is_peak_hour":      1 if hour in range(7, 11) else 0,
        "hour_sin":          np.sin(2*np.pi*hour/24),
        "hour_cos":          np.cos(2*np.pi*hour/24),
        "dow_sin":           np.sin(2*np.pi*day_of_week/7),
        "dow_cos":           np.cos(2*np.pi*day_of_week/7),
        "port_traffic":      td["port_traffic"],
        "traffic_3h":        td["traffic_3h"],
        "traffic_6h":        td["traffic_6h"],
        "prev_traffic":      td["prev_traffic"],
        "vessel_avg_delay":  vd.get(vessel, VESSEL_TYPICAL.get(vessel, -8.0)),
        "prev_dep_delay":    prev_dep_delay,
        "rolling_delay_3v":  rolling_delay,
        "prev_was_late":     prev_was_late,
        "is_hel_to_tal":     is_hel_to_tal,
        "sched_duration_min":sched_duration,
    }
    return pd.DataFrame([row])[MODEL_FEATURES]

def rule_prob(vessel, hour, dow, month):
    """Fallback rule-based delay probability when no model is available."""
    base   = {"Megastar": 0.06, "Star": 0.08, "Finlandia": 0.12, "Europa": 0.14}
    prob   = base.get(vessel, 0.10)
    prob  += 0.05 if hour in range(8, 20) else -0.02  # daytime busier
    prob  += 0.04 if dow >= 5 else 0.0                # weekend
    prob  += 0.03 if month in [1, 2, 12] else 0.0     # winter months
    return min(0.95, max(0.01, prob))


df              = load_data()
bundle          = load_model_bundle()
model           = bundle["model"]          if bundle else None
threshold       = bundle.get("threshold", 0.40) if bundle else 0.40
metrics         = load_json("outputs/model_metrics.json")
anomaly_report  = load_json("outputs/anomaly_report.json")
cluster_metrics = load_json("outputs/cluster_metrics.json")
opt_report      = load_json("outputs/optimization_report.json")
forecast_metrics= load_json("outputs/forecast_metrics.json")
forecast_df     = load_forecast()
schedule_df     = load_schedule()
clusters_df     = load_clusters()
net_data        = load_json("outputs/port_network.json")
td              = compute_traffic_defaults(df)
vd              = compute_vessel_avg_delay(df)


def H(t, s=""):
    st.markdown(f'<div class="pc-title">{t}</div><div class="pc-sub">{s}</div>', unsafe_allow_html=True)

def SEC(t):
    st.markdown(f'<div class="pc-sec">{t}</div>', unsafe_allow_html=True)

def CARD(html, cls=""):
    st.markdown(f'<div class="pc-card {cls}">{html}</div>', unsafe_allow_html=True)

def BADGE(text, cls="cy"):
    return f'<span class="pc-badge {cls}">{text}</span>'

def risk_box(prob, rule_based=False):
    pct = f"{prob:.1%}"
    if prob > 0.50:   c, icon, lbl = "rd", "🚨", "HIGH RISK"
    elif prob > 0.25: c, icon, lbl = "am", "⚠️", "MODERATE RISK"
    else:             c, icon, lbl = "gr", "✅", "LOW RISK"
    tag = " · rule-based" if rule_based else ""
    col_map = {"rd": "red", "am": "amber", "gr": "green"}
    return (f'<div class="pc-rb" style="border-top:3px solid var(--{col_map[c]})">'
            f'<div class="pc-rl">{icon} Delay Probability{tag}</div>'
            f'<div class="pc-rv" style="color:var(--{col_map[c]})">{pct}</div>'
            f'<div class="pc-rs">{lbl}</div></div>')

def fi_bars(fi_dict):
    if not fi_dict:
        return ""
    fi = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)
    mx = fi[0][1] if fi else 1
    html = ""
    for feat, imp in fi:
        pct = imp/mx*100
        html += (f'<div style="margin-bottom:8px">'
                 f'<div style="display:flex;justify-content:space-between;font-family:monospace;font-size:.78rem;margin-bottom:3px">'
                 f'<span style="color:#94a3b8">{feat}</span><span style="color:#00d4ff">{imp:.4f}</span></div>'
                 f'<div style="background:#1e2d45;border-radius:2px;height:5px">'
                 f'<div style="background:linear-gradient(90deg,#00d4ff,#0099bb);width:{pct:.1f}%;height:5px;border-radius:2px"></div></div></div>')
    return html


with st.sidebar:
    st.markdown('<div style="font-family:monospace;font-size:1.1rem;font-weight:700;color:#00d4ff">⚓ PORT COMMANDER</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:.72rem;color:#475569;margin-bottom:1.2rem">Helsinki ↔ Tallinn · AIS 2018-19</div>', unsafe_allow_html=True)

    n_v = f"{len(df):,} voyages" if df is not None else "run phase1.py"
    statuses = [
        ("Dataset",    df is not None,              n_v),
        ("ML Model",   model is not None,            "ready" if model else "run phase3.py"),
        ("Clusters",   clusters_df is not None,      "ready" if clusters_df is not None else "run phase2_clustering.py"),
        ("Anomaly",    anomaly_report is not None,   "ready" if anomaly_report else "run phase2_anomaly.py"),
        ("Optimizer",  opt_report is not None,       "ready" if opt_report else "run phase3_optimizer.py"),
        ("Forecast",   forecast_df is not None,      "ready" if forecast_df is not None else "run phase4_forecast.py"),
    ]
    for name, ok, note in statuses:
        col = "#10b981" if ok else "#ef4444"
        bg  = "rgba(16,185,129,.08)" if ok else "rgba(239,68,68,.08)"
        st.markdown(
            f'<div style="background:{bg};border:1px solid {col}33;border-radius:5px;'
            f'padding:5px 10px;margin-bottom:4px;font-size:.73rem;color:{col}">'
            f'{"✓" if ok else "✗"}  <b>{name}</b> · {note}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    page = st.radio("", [
        "📊  Overview",
        "🔮  Delay Predictor",
        "📅  Schedule Optimizer",
        "🔬  Clustering & Network",
        "🚨  Anomaly Detection",
        "⚙️  Berth Optimizer",
        "📡  Congestion Forecast",
        "📈  Model Performance",
    ], label_visibility="collapsed")

    st.markdown('<div style="font-family:monospace;font-size:.65rem;color:#334155;text-align:center;margin-top:2rem">v4.0 · Port Commander</div>', unsafe_allow_html=True)



if "Overview" in page:
    H("Corridor Overview", "Live statistics from AIS data — Helsinki ↔ Tallinn ferry corridor (2018–2019)")

    if df is None:
        st.error("voyage_dataset.csv not found. Run phase1.py first.")
        st.stop()

    total    = len(df)
    delayed  = int(df["is_delayed"].sum()) if "is_delayed" in df.columns else 0
    rate     = delayed / total * 100
    avg_dep  = float(df["dep_delay_min"].mean()) if "dep_delay_min" in df.columns else 0
    vessels  = df["ship"].nunique() if "ship" in df.columns else 0
    n_anom   = anomaly_report["combined_anomalies"] if anomaly_report else 0
    date_rng = f"{df['etd_sched'].min().strftime('%b %Y')} – {df['etd_sched'].max().strftime('%b %Y')}" if "etd_sched" in df.columns else "—"

    SEC("Fleet KPIs")
    k = st.columns(6)
    k[0].metric("Total Voyages",      f"{total:,}")
    k[1].metric("Delay Rate",         f"{rate:.1f}%",    delta=f"{rate-10:.1f}pp vs 10% baseline", delta_color="inverse")
    k[2].metric("On-Time Rate",       f"{100-rate:.1f}%")
    k[3].metric("Avg Dep Delay",      f"{avg_dep:+.1f} min")
    k[4].metric("Active Vessels",     str(vessels))
    k[5].metric("Anomalies Detected", str(n_anom))

    st.caption(f"Dataset period: {date_rng}")
    st.markdown("---")

    col_a, col_b = st.columns([3, 2], gap="large")

    with col_a:
        SEC("Departure Delay by Vessel")
        if "dep_delay_min" in df.columns and "ship" in df.columns:
            vs = (df.groupby("ship")["dep_delay_min"]
                  .agg(["mean", "std", "count"])
                  .round(2)
                  .rename(columns={"mean": "Avg Delay (min)", "std": "Std Dev", "count": "Voyages"})
                  .sort_values("Avg Delay (min)"))
            st.dataframe(vs, use_container_width=True)

            cols = st.columns(len(vs))
            for i, (vessel, row) in enumerate(vs.iterrows()):
                v = row["Avg Delay (min)"]
                c = "gr" if v < -5 else "am" if v < 5 else "rd"
                l = "Early" if v < -5 else "On Time" if v < 5 else "Late"
                with cols[i]:
                    st.markdown(
                        f'<div style="text-align:center">'
                        f'<div style="font-family:monospace;font-size:.8rem;color:#94a3b8;margin-bottom:3px">{vessel}</div>'
                        f'{BADGE(f"{v:+.1f}m · {l}", c)}</div>',
                        unsafe_allow_html=True,
                    )

    with col_b:
        SEC("Delay Class Distribution")
        if "delay_class" in df.columns:
            names  = {0: "Very Early", 1: "Early", 2: "On Time", 3: "Late", 4: "Very Late"}
            counts = df["delay_class"].value_counts().sort_index()
            cls_df = pd.DataFrame({
                "Status":  [names.get(i, str(i)) for i in counts.index],
                "Voyages": counts.values,
                "Share":   [f"{v/total*100:.1f}%" for v in counts.values],
            })
            st.dataframe(cls_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    SEC("Congestion Heatmap · Avg Departure Delay (min) by Vessel × Departure Hour")
    if "dep_delay_min" in df.columns and "hour_of_dep" in df.columns:
        hm = (df.groupby(["ship", "hour_of_dep"])["dep_delay_min"]
              .mean()
              .unstack("hour_of_dep")
              .round(1)
              .reindex(columns=range(24))
              .fillna(0))
        st.dataframe(
            hm.style.background_gradient(cmap="RdYlGn_r", axis=None).format("{:+.1f}"),
            use_container_width=True,
        )
        st.caption("Values in minutes. Red = delayed, Green = early. Missing hours = no voyages in dataset.")

    st.markdown("---")
    SEC("Vessel Statistics by Route Direction")
    if "is_hel_to_tal" in df.columns:
        route_stats = df.groupby(["ship", "is_hel_to_tal"]).agg(
            voyages   = ("ship",          "count"),
            avg_delay = ("dep_delay_min", "mean"),
            pct_late  = ("is_delayed",    "mean"),
            avg_speed = ("mean_sog",      "mean"),
        ).round(2).reset_index()
        route_stats["direction"] = route_stats["is_hel_to_tal"].map({1: "HEL→TLL", 0: "TLL→HEL"})
        st.dataframe(route_stats.drop(columns="is_hel_to_tal"), use_container_width=True, hide_index=True)

    st.markdown("---")
    SEC("Key Insights from Dataset")
    if "dep_delay_min" in df.columns and "hour_of_dep" in df.columns:
        hmean   = df.groupby("hour_of_dep")["dep_delay_min"].mean()
        worst_h = int(hmean.idxmax())
        best_h  = int(hmean.idxmin())
        best_v  = df.groupby("ship")["dep_delay_min"].std().idxmin()
        worst_v = df.groupby("ship")["dep_delay_min"].mean().idxmax()
        peak_h  = anomaly_report.get("peak_anomaly_hour") if anomaly_report else None
        most_v  = df.groupby("ship")["ship"].count().idxmax()

        i1, i2, i3, i4, i5 = st.columns(5)
        for col, icon, lbl, val, desc in [
            (i1, "⏰", "Worst Dep Hour",   f"{worst_h:02d}:00", f"avg {hmean[worst_h]:+.1f} min"),
            (i2, "✅", "Best Dep Hour",    f"{best_h:02d}:00",  f"avg {hmean[best_h]:+.1f} min"),
            (i3, "🏆", "Most Reliable",    best_v,               "lowest delay std"),
            (i4, "📌", "Most Departures",  most_v,               f"{df['ship'].value_counts()[most_v]} voyages"),
            (i5, "🌩", "Peak Anomaly Hr",  f"{peak_h:02d}:00" if peak_h is not None else "N/A", "most disruptions"),
        ]:
            with col:
                st.markdown(
                    f'<div class="pc-ib"><div style="font-size:1.4rem">{icon}</div>'
                    f'<div style="font-size:.68rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#475569">{lbl}</div>'
                    f'<div style="font-family:monospace;font-size:1.1rem;color:#e2e8f0;margin-top:.15rem">{val}</div>'
                    f'<div style="font-size:.75rem;color:#475569;margin-top:.1rem">{desc}</div></div>',
                    unsafe_allow_html=True,
                )


elif "Predictor" in page:
    H("Delay Predictor", "Enter voyage parameters to estimate departure delay risk")
    if not model:
        st.warning("⚠️ No ML model found — run `python phase3.py`. Showing rule-based estimates.")

    SEC("Voyage Parameters")
    vessels_available = sorted(df["ship"].unique()) if df is not None else list(VESSEL_TYPICAL.keys())
    c1, c2, c3, c4 = st.columns(4)
    with c1: vessel = st.selectbox("Vessel", vessels_available)
    with c2: day = st.selectbox("Day of Week", list(DAY_MAP.keys())); dow = DAY_MAP[day]
    with c3: month = st.selectbox("Month", list(range(1, 13)), format_func=lambda m: MONTH_NAMES[m-1], index=5)
    with c4: hour_str = st.selectbox("Departure Hour", [f"{h:02d}:00" for h in range(24)], index=8); hour = int(hour_str[:2])

    mean_sog     = st.slider("Expected Speed (knots)", 10.0, 27.0, 21.0, 0.5)
    direction    = st.radio("Route Direction", ["HEL → TLL", "TLL → HEL"], horizontal=True)
    is_hel2tal   = 1 if "HEL" in direction else 0

    with st.expander("Advanced parameters (vessel history)"):
        c5, c6, c7 = st.columns(3)
        with c5: prev_delay  = st.slider("Previous voyage delay (min)", -30.0, 30.0, 0.0, 0.5)
        with c6: roll_delay  = st.slider("Rolling 3-voyage avg delay (min)", -20.0, 20.0, -8.0, 0.5)
        with c7: was_late    = st.checkbox("Previous voyage was late (>5 min)", value=False)

    sched_dur = 120.0  

    if st.button("  🚀  Run Prediction  ", type="primary"):
        inp    = build_feature_row(vessel, hour, dow, month, mean_sog, td, vd,
                                   prev_delay, roll_delay, int(was_late), is_hel2tal, sched_dur)
        rb     = model is None
        if model:
            try:
                proba    = model.predict_proba(inp)[0][1]
                pred_cls = int(proba >= threshold)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.stop()
        else:
            proba    = rule_prob(vessel, hour, dow, month)
            pred_cls = int(proba >= 0.40)

        st.markdown("---")
        SEC("Prediction Results")
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(risk_box(proba, rb), unsafe_allow_html=True)
        with r2:
            cc = "var(--red)" if pred_cls == 1 else "var(--green)"
            st.markdown(
                f'<div class="pc-rb" style="border-top:3px solid {cc}">'
                f'<div class="pc-rl">Prediction (thresh={threshold:.2f})</div>'
                f'<div class="pc-rv" style="color:{cc};font-size:1.5rem">{"DELAYED" if pred_cls else "ON TIME"}</div>'
                f'<div class="pc-rs">{"ML model" if model else "Rule-based"}</div></div>',
                unsafe_allow_html=True,
            )
        with r3:
            typ = vd.get(vessel, VESSEL_TYPICAL.get(vessel, -8.0))
            st.markdown(
                f'<div class="pc-rb" style="border-top:3px solid var(--cyan)">'
                f'<div class="pc-rl">{vessel} Historical Avg Delay</div>'
                f'<div class="pc-rv" style="color:var(--cyan);font-size:1.5rem">{typ:+.1f} min</div>'
                f'<div class="pc-rs">Computed from dataset</div></div>',
                unsafe_allow_html=True,
            )

        with st.expander("🔍 Feature vector submitted to model"):
            st.dataframe(inp.T.rename(columns={0: "Value"}).round(4), use_container_width=True)



elif "Schedule Optimizer" in page:
    H("Schedule Optimizer", "Scan all 24 departure hours to find the lowest-risk slots for a vessel")
    if not model:
        st.warning("⚠️ No ML model — showing rule-based risk estimates.")

    SEC("Parameters")
    vessels_available = sorted(df["ship"].unique()) if df is not None else list(VESSEL_TYPICAL.keys())
    c1, c2, c3 = st.columns(3)
    with c1: vessel_opt = st.selectbox("Vessel", vessels_available, key="sched_vessel")
    with c2: day_opt = st.selectbox("Day", list(DAY_MAP.keys()), key="sched_day"); dow_opt = DAY_MAP[day_opt]
    with c3: month_opt = st.selectbox("Month", list(range(1, 13)), format_func=lambda m: MONTH_NAMES[m-1], index=5, key="sched_month")
    sog_opt   = st.slider("Expected Speed (knots)", 10.0, 27.0, 21.0, 0.5, key="sched_sog")
    dir_opt   = st.radio("Route Direction", ["HEL → TLL", "TLL → HEL"], horizontal=True, key="sched_dir")
    hel2tal   = 1 if "HEL" in dir_opt else 0

    if st.button("  🔍  Scan All Hours  ", type="primary"):
        with st.spinner("Scanning 24 departure slots…"):
            results = []
            for h in range(24):
                inp = build_feature_row(vessel_opt, h, dow_opt, month_opt, sog_opt, td, vd,
                                        prev_dep_delay=vd.get(vessel_opt, -8.0),
                                        is_hel_to_tal=hel2tal)
                if model:
                    try:
                        prob = model.predict_proba(inp)[0][1]
                    except Exception:
                        prob = rule_prob(vessel_opt, h, dow_opt, month_opt)
                else:
                    prob = rule_prob(vessel_opt, h, dow_opt, month_opt)

                results.append({
                    "Hour":   f"{h:02d}:00",
                    "Risk":   "🟢 Low" if prob < 0.20 else "🟡 Moderate" if prob < 0.50 else "🔴 High",
                    "Risk %": f"{prob:.1%}",
                    "Action": "✅ Depart" if prob < 0.20 else "⚠️ Caution" if prob < 0.50 else "🚫 Avoid",
                    "_p":     prob,
                })

        st.markdown("---")
        SEC(f"Results — {vessel_opt} · {day_opt} · {dir_opt}")
        safe  = sorted([r for r in results if "✅" in r["Action"]], key=lambda x: x["_p"])
        risky = sorted([r for r in results if "🚫" in r["Action"]], key=lambda x: x["_p"], reverse=True)

        if safe:
            top3 = " · ".join(f'<b>{s["Hour"]}</b> ({s["Risk %"]})' for s in safe[:3])
            CARD(f'{BADGE("RECOMMENDED","gr")} <span style="font-size:.85rem;color:#94a3b8;margin-left:10px">Best slots: {top3}</span>', "pc-card-gr")
        if risky:
            avoid3 = " · ".join(f'<b>{r["Hour"]}</b>' for r in risky[:3])
            CARD(f'{BADGE("AVOID","rd")} <span style="font-size:.85rem;color:#94a3b8;margin-left:10px">High-risk: {avoid3}</span>', "pc-card-rd")

        res_df = pd.DataFrame(results)[["Hour", "Risk", "Risk %", "Action"]]
        st.dataframe(res_df, use_container_width=True, hide_index=True)

        # Show actual historical delay for this vessel by hour
        if df is not None and vessel_opt in df["ship"].values:
            SEC(f"Historical Data: {vessel_opt} Avg Delay by Hour (from dataset)")
            hist_h = (df[df["ship"] == vessel_opt]
                      .groupby("hour_of_dep")["dep_delay_min"]
                      .agg(["mean", "count"])
                      .round(2)
                      .rename(columns={"mean": "Avg Delay", "count": "Voyages"}))
            hist_h.index = [f"{h:02d}:00" for h in hist_h.index]
            st.dataframe(hist_h, use_container_width=True)



elif "Clustering" in page:
    H("Clustering & Network Analysis", "Vessel behaviour clusters, port bottleneck detection, and flow graph")

    if not cluster_metrics:
        st.warning("Run `python phase2_clustering.py` first.")
        st.stop()

    SEC("Clustering Summary")
    kc = st.columns(4)
    kc[0].metric("Vessel Clusters",  str(cluster_metrics.get("best_k", "—")))
    kc[1].metric("Silhouette Score", f'{cluster_metrics.get("silhouette_score", 0):.3f}')
    kc[2].metric("Ports in Network", str(cluster_metrics.get("n_ports", "—")))
    kc[3].metric("Routes Mapped",    str(cluster_metrics.get("n_routes", "—")))

    st.markdown("---")
    ca, cb = st.columns(2, gap="large")
    with ca:
        SEC("Vessel Cluster Assignments")
        if clusters_df is not None:
            show_cols = [c for c in ["ship", "cluster", "cluster_label",
                                      "avg_dep_delay", "pct_delayed", "avg_speed", "total_voyages"]
                         if c in clusters_df.columns]
            st.dataframe(clusters_df[show_cols].round(3), use_container_width=True, hide_index=True)

    with cb:
        SEC("Cluster Profiles")
        if cluster_metrics.get("vessel_clusters"):
            cp_df = pd.DataFrame(cluster_metrics["vessel_clusters"])
            st.dataframe(cp_df.round(3), use_container_width=True, hide_index=True)
            for _, row in cp_df.iterrows():
                lbl   = row.get("label", "")
                delay = row.get("avg_dep_delay", 0)
                cls   = "rd" if lbl == "High-Risk" else "gr" if lbl == "Early-Consistent" else "cy" if lbl == "Reliable" else "am"
                st.markdown(BADGE(f'Cluster {int(row["cluster"])} · {lbl} · avg {delay:+.1f}min', cls) + "&nbsp;", unsafe_allow_html=True)

    st.markdown("---")
    SEC("Port Bottleneck Analysis (NetworkX)")
    if net_data:
        nodes_df = pd.DataFrame(net_data["nodes"])
        edges_df = pd.DataFrame(net_data["edges"])
        n1, n2 = st.columns(2, gap="large")
        with n1:
            st.markdown("**Port Nodes** — sorted by betweenness centrality")
            st.dataframe(nodes_df.sort_values("betweenness", ascending=False).round(3),
                         use_container_width=True, hide_index=True)
        with n2:
            st.markdown("**Route Edges** — voyages and avg delay per direction")
            st.dataframe(edges_df.round(3), use_container_width=True, hide_index=True)

        bottlenecks = cluster_metrics.get("bottleneck_ports", [])
        if bottlenecks:
            st.markdown("---")
            SEC("Top Bottleneck Ports")
            for b in bottlenecks[:5]:
                score = b["betweenness"]
                cls   = "rd" if score > 0.5 else "am" if score > 0.2 else "cy"
                st.markdown(BADGE(f'{b["port"]} · betweenness {score:.4f}', cls) + "&nbsp;", unsafe_allow_html=True)

    st.markdown("---")
    if cluster_metrics.get("hour_congestion"):
        SEC("Hour-Block Congestion Tiers")
        hc_df = pd.DataFrame(cluster_metrics["hour_congestion"])
        st.dataframe(hc_df.round(3), use_container_width=True, hide_index=True)
        tier_map = {"Low Congestion": "gr", "Medium Congestion": "am", "High Congestion": "rd"}
        for label, cls in tier_map.items():
            hrs = [str(r["hour"]) for r in cluster_metrics["hour_congestion"] if r.get("congestion_label") == label]
            if hrs:
                st.markdown(BADGE(f'{label}: hours {", ".join(hrs)}', cls) + "&nbsp;", unsafe_allow_html=True)


elif "Anomaly" in page:
    H("Anomaly Detection", "Isolation Forest + Z-score outlier analysis and weather-proxy delay spikes")

    if not anomaly_report:
        st.warning("Run `python phase2_anomaly.py` first.")
        st.stop()

    ar = anomaly_report
    SEC("Anomaly Summary")
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Total Voyages",       f'{ar["total_voyages"]:,}')
    a2.metric("Isolation Forest",    str(ar["iso_anomalies"]),    delta=f'{ar["iso_anomalies"]/ar["total_voyages"]:.1%}')
    a3.metric("Z-Score Anomalies",   str(ar["zscore_anomalies"]), delta=f'{ar["zscore_anomalies"]/ar["total_voyages"]:.1%}')
    a4.metric("Combined Flag",       str(ar["combined_anomalies"]))
    a5.metric("Top Anomaly Vessel",  str(ar.get("top_anomaly_vessel", "N/A")))

    st.markdown("---")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        SEC("Weather Proxy Analysis")
        CARD(
            f'<div style="font-size:.78rem;color:#94a3b8;margin-bottom:.8rem">'
            f'Proxy derived from low crossing speed + high departure delay</div>'
            f'<table style="width:100%;font-size:.82rem;border-collapse:collapse">'
            f'<tr><td style="padding:6px 0;border-bottom:1px solid #1e2d45;color:#94a3b8">Adverse weather % of voyages</td>'
            f'<td style="font-family:monospace;color:#00d4ff">{ar["adverse_weather_pct"]:.1%}</td></tr>'
            f'<tr><td style="padding:6px 0;border-bottom:1px solid #1e2d45;color:#94a3b8">Avg delay — adverse conditions</td>'
            f'<td style="font-family:monospace;color:#ef4444">{ar["avg_delay_adverse"]:+.1f} min</td></tr>'
            f'<tr><td style="padding:6px 0;color:#94a3b8">Avg delay — normal conditions</td>'
            f'<td style="font-family:monospace;color:#10b981">{ar["avg_delay_normal"]:+.1f} min</td></tr>'
            f'</table>', "pc-card-am"
        )

    with c2:
        SEC("Weather-Linked Delay Spikes")
        weather_impact = pd.DataFrame(ar.get("weather_impact", []))
        if not weather_impact.empty:
            weather_impact["Condition"] = weather_impact["adverse_weather_flag"].map({0: "Normal", 1: "Adverse"})
            show = weather_impact[["Condition", "avg_delay", "pct_delayed", "avg_speed", "n_voyages"]].round(3)
            st.dataframe(show, use_container_width=True, hide_index=True)
        spikes = ar.get("weather_delay_spikes", 0)
        CARD(
            f'{BADGE(f"{spikes} weather-linked spikes","am")} '
            f'<span style="font-size:.82rem;color:#94a3b8;margin-left:8px">'
            f'({spikes/max(ar["total_voyages"],1):.1%} of all voyages)</span>',
            "pc-card-am"
        )

    st.markdown("---")
    SEC("Detection Method Comparison")
    methods_df = pd.DataFrame({
        "Method":       ["Isolation Forest", "Z-Score (|z|>3)", "Combined Flag", "Weather Proxy Spikes"],
        "Flags":        [ar["iso_anomalies"], ar["zscore_anomalies"],
                         ar["combined_anomalies"], ar.get("weather_delay_spikes", 0)],
        "% of Dataset": [f'{ar["iso_anomalies"]/ar["total_voyages"]:.1%}',
                         f'{ar["zscore_anomalies"]/ar["total_voyages"]:.1%}',
                         f'{ar["combined_anomalies"]/ar["total_voyages"]:.1%}',
                         f'{ar.get("weather_delay_spikes",0)/ar["total_voyages"]:.1%}'],
        "Use":          ["Multi-feature outlier", "Single-feature extreme",
                         "Broad safety net",     "Operational risk flag"],
    })
    st.dataframe(methods_df, use_container_width=True, hide_index=True)

    if df is not None and "is_anomaly" in df.columns:
        st.markdown("---")
        SEC("Anomalous Voyage Records (sample)")
        anom_df = df[df["is_anomaly"] == 1][
            ["ship", "etd_sched", "dep_delay_min", "mean_sog", "port_traffic", "weather_proxy_score"]
        ].head(20)
        st.dataframe(anom_df.round(3), use_container_width=True, hide_index=True)



elif "Berth" in page:
    H("Berth & Schedule Optimizer", "Constraint-based schedule optimisation with vessel priority and capacity limits")

    if not opt_report:
        st.warning("Run `python phase3_optimizer.py` first.")
        st.stop()

    r = opt_report
    SEC("Optimisation Results")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Voyages Rescheduled",  f'{r["voyages_rescheduled"]} ({r["pct_rescheduled"]:.0%})')
    o2.metric("Avg Delay Before",     f'{r["avg_delay_before_min"]:+.1f} min')
    o3.metric("Avg Delay After",      f'{r["avg_delay_after_min"]:+.1f} min',
              delta=f'{r["delay_reduction_min"]:+.1f} min saved', delta_color="normal")
    o4.metric("Method",               r["optimizer_method"])

    st.markdown("---")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        SEC("Cost Savings Estimate")
        CARD(
            f'<div style="font-size:.78rem;color:#94a3b8;margin-bottom:.8rem">'
            f'Based on vessel priority × delay cost rates (€/min)</div>'
            f'<table style="width:100%;font-size:.85rem;border-collapse:collapse">'
            f'<tr><td style="padding:8px 0;border-bottom:1px solid #1e2d45;color:#94a3b8">Cost BEFORE</td>'
            f'<td style="font-family:monospace;color:#ef4444">€{r["cost_before_eur"]:,.0f}</td></tr>'
            f'<tr><td style="padding:8px 0;border-bottom:1px solid #1e2d45;color:#94a3b8">Cost AFTER</td>'
            f'<td style="font-family:monospace;color:#10b981">€{r["cost_after_eur"]:,.0f}</td></tr>'
            f'<tr><td style="padding:8px 0;color:#94a3b8">Saving</td>'
            f'<td style="font-family:monospace;color:#00d4ff">€{r["cost_saving_eur"]:,.0f} ({r["cost_saving_pct"]:.1f}%)</td></tr>'
            f'</table>', "pc-card-gr"
        )
        SEC("Departure Hour Recommendations")
        best  = r.get("best_departure_hours", [])
        worst = r.get("worst_departure_hours", [])
        if best:
            st.markdown(BADGE("Best Hours", "gr") + " " + " ".join(BADGE(h, "gr") for h in best), unsafe_allow_html=True)
        if worst:
            st.markdown("<br>" + BADGE("Avoid Hours", "rd") + " " + " ".join(BADGE(h, "rd") for h in worst), unsafe_allow_html=True)

    with c2:
        SEC("Rerouting Recommendations")
        recs = r.get("rerouting_recommendations", [])
        if recs:
            rec_df = pd.DataFrame(recs)[["vessel", "priority", "affected_voyages", "potential_saving_min", "rec_hours"]]
            rec_df["rec_hours"] = rec_df["rec_hours"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
            st.dataframe(rec_df.round(1), use_container_width=True, hide_index=True)
        CARD(
            f'<div style="font-size:.78rem;color:#94a3b8;margin-bottom:.6rem">Constraint parameters</div>'
            f'<div style="font-family:monospace;font-size:.78rem;color:#94a3b8">'
            f'Port capacity  : {r.get("port_capacities", {})}<br>'
            f'Max shift      : ±2.0h<br>'
            f'Min vessel gap : 1.5h</div>', "pc-card-cy"
        )

    if schedule_df is not None:
        st.markdown("---")
        SEC("Rescheduled Voyages (sample of changed voyages)")
        show_cols = [c for c in ["ship", "depPort", "arrPort", "etd_sched", "optimized_etd",
                                  "shift_hours", "shift_reason", "dep_delay_min", "opt_delay_est"]
                     if c in schedule_df.columns]
        changed = schedule_df[schedule_df["shift_hours"] != 0] if "shift_hours" in schedule_df.columns else schedule_df
        st.dataframe(changed[show_cols].head(30).round(2), use_container_width=True, hide_index=True)
        st.caption(f"Showing {min(30, len(changed))} rescheduled voyages out of {len(schedule_df):,} total")


elif "Forecast" in page:
    H("Congestion Forecast", "Time-series prediction of daily average departure delay for the next 30 days")

    if not forecast_metrics:
        st.warning("Run `python phase4_forecast.py` first.")
        st.stop()

    fm = forecast_metrics
    SEC("Forecast Summary")
    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric("Method",              fm["method"])
    f2.metric("Horizon",             f'{fm["forecast_horizon_days"]} days')
    f3.metric("Holdout MAE",         f'{fm["holdout_mae_min"]:.2f} min' if fm.get("holdout_mae_min") else "N/A")
    f4.metric("Avg Forecast Delay",  f'{fm["avg_forecast_delay"]:+.1f} min')
    f5.metric("High-Risk Days Ahead",str(fm["peak_days_count"]))

    st.markdown("---")
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        SEC("14-Day Forecast")
        if fm.get("future_forecast_sample"):
            fc_df = pd.DataFrame(fm["future_forecast_sample"])
            fc_df.columns = ["Date", "Forecast (min)", "Lower CI (90%)", "Upper CI (90%)"]
            st.dataframe(
                fc_df.style.background_gradient(subset=["Forecast (min)"], cmap="RdYlGn_r"),
                use_container_width=True, hide_index=True,
            )

    with c2:
        SEC("Peak Congestion Days")
        peaks = fm.get("peak_days", [])
        if peaks:
            pk_df = pd.DataFrame(peaks)
            pk_df.columns = ["Date", "Forecast Delay (min)"]
            st.dataframe(pk_df, use_container_width=True, hide_index=True)
            thresh = fm.get("peak_congestion_threshold", 0)
            CARD(
                f'{BADGE(f"{len(peaks)} high-risk days","rd")} '
                f'<span style="font-size:.82rem;color:#94a3b8;margin-left:8px">'
                f'threshold: {thresh:.1f} min</span>',
                "pc-card-rd"
            )
        else:
            st.info("No peak days identified in forecast period.")

    st.markdown("---")
    SEC("Monthly Congestion Trend (Actual Data)")
    if fm.get("monthly_trend"):
        mt_df = pd.DataFrame(fm["monthly_trend"])
        mt_df.columns = ["Month", "Avg Delay (min)", "% Delayed", "Voyages"]
        st.dataframe(
            mt_df.style.background_gradient(subset=["Avg Delay (min)"], cmap="RdYlGn_r"),
            use_container_width=True, hide_index=True,
        )
        st.caption("Actual aggregated data from voyage_dataset.csv — not forecasted values.")

    if forecast_df is not None:
        st.markdown("---")
        SEC("Full Forecast Series (last 60 rows)")
        show_fc = forecast_df.dropna(subset=["forecast_delay"]).tail(60)
        display_cols = [c for c in ["date", "avg_delay", "forecast_delay", "lower_ci", "upper_ci"] if c in show_fc.columns]
        st.dataframe(show_fc[display_cols].round(2), use_container_width=True, hide_index=True)
        st.caption(f"Method: {fm['method']} · Train: {fm['train_days']} days · Test: {fm['test_days']} days")


elif "Performance" in page:
    H("Model Performance", "Evaluation metrics, CV results, feature importances, and leakage audit")

    if not metrics:
        st.info("Run `python phase3.py` to generate model metrics.")
        st.stop()

    SEC("Test Set Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",   f'{metrics.get("accuracy",  0):.1%}')
    m2.metric("Precision",  f'{metrics.get("precision", 0):.1%}')
    m3.metric("Recall",     f'{metrics.get("recall",    0):.1%}', help="% of actual delays caught")
    m4.metric("F1 Score",   f'{metrics.get("f1",        0):.1%}')
    m5.metric("ROC-AUC",    f'{metrics.get("roc_auc",   0):.3f}' if metrics.get("roc_auc") else "N/A")

    st.markdown("---")
    SEC("Overfitting Check")
    d1, d2, d3 = st.columns(3)
    ta  = metrics.get("train_accuracy", 0)
    te  = metrics.get("test_accuracy",  0)
    gap = ta - te
    d1.metric("Train Accuracy", f'{ta:.1%}')
    d2.metric("Test Accuracy",  f'{te:.1%}')
    d3.metric("Train-Test Gap", f'{gap:.1%}',
              delta="Overfit" if gap > 0.10 else "OK",
              delta_color="inverse" if gap > 0.10 else "normal")

    st.markdown("---")
    SEC("Threshold Analysis")
    if metrics.get("threshold_analysis"):
        ta_rows = []
        for thresh, tv in metrics["threshold_analysis"].items():
            ta_rows.append({"Threshold": thresh, **{k: f"{v:.3f}" for k, v in tv.items()}})
        thresh_used = metrics.get("threshold", "0.4")
        st.dataframe(pd.DataFrame(ta_rows), use_container_width=True, hide_index=True)
        st.caption(f"✅ Chosen threshold: {thresh_used} (maximises F1 on test set)")

    st.markdown("---")
    SEC("Cross-Validation Results (train set)")
    if metrics.get("cv_results"):
        cv_rows = [{"Model": k, **{m: f"{v:.4f}" for m, v in vs.items()}} for k, vs in metrics["cv_results"].items()]
        st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)
        st.caption("5-fold StratifiedKFold on train set. No shuffle — respects time ordering.")

    if metrics.get("feature_importance"):
        st.markdown("---")
        SEC("Feature Importances")
        CARD(fi_bars(metrics["feature_importance"]))

    st.markdown("---")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        SEC("Why Recall > Accuracy for Port Operations")
        CARD(
            '<table style="width:100%;font-size:.8rem;border-collapse:collapse">'
            '<tr><td style="padding:6px 0;border-bottom:1px solid #1e2d45;color:#ef4444;font-weight:600">Miss a delay (FN)</td>'
            '<td style="padding:6px 0;border-bottom:1px solid #1e2d45;color:#94a3b8">Berth conflict, vessel waits at anchor, domino delays</td></tr>'
            '<tr><td style="padding:6px 0;color:#f59e0b;font-weight:600">False alarm (FP)</td>'
            '<td style="padding:6px 0;color:#94a3b8">Minor berth re-planning, low operational cost</td></tr>'
            '</table>'
            '<div style="font-size:.75rem;color:#475569;margin-top:.6rem">Missing a delay ≈ 10× more costly than a false alarm. '
            'Lower decision threshold (0.30–0.40) boosts recall.</div>',
            "pc-card-cy"
        )
    with c2:
        SEC("Data Leakage Audit")
        audit_df = pd.DataFrame({
            "Risk":    ["vessel_avg_delay on full data", "Traffic rolling before split",
                        "arr_delay_min in features",     "Target in features"],
            "Status":  ["✅ Fixed — recomputed on train split only",
                        "✅ Fixed — chronological split preserves order",
                        "✅ Excluded — used only for is_delayed target",
                        "✅ Excluded"],
        })
        st.dataframe(audit_df, use_container_width=True, hide_index=True)

    SEC("Feature Reference")
    feat_desc = {
        "mean_sog":          "Avg crossing speed (kn)",
        "is_slow_crossing":  "1 if mean_sog < 15 kn",
        "hour_of_dep":       "Departure hour (0–23)",
        "day_of_week":       "Day 0=Mon…6=Sun",
        "month":             "Month 1–12",
        "is_weekend":        "1 if Sat/Sun",
        "is_peak_hour":      "1 if 07–10h",
        "hour_sin":          "Cyclical: sin(hour/24·2π)",
        "hour_cos":          "Cyclical: cos(hour/24·2π)",
        "dow_sin":           "Cyclical: sin(dow/7·2π)",
        "dow_cos":           "Cyclical: cos(dow/7·2π)",
        "port_traffic":      "Vessels departing same hour",
        "traffic_3h":        "Rolling 3h avg traffic",
        "traffic_6h":        "Rolling 6h avg traffic",
        "prev_traffic":      "Traffic in previous period",
        "vessel_avg_delay":  "Vessel's avg dep delay (train-only)",
        "prev_dep_delay":    "Vessel's last voyage dep delay",
        "rolling_delay_3v":  "Vessel 3-voyage rolling dep delay",
        "prev_was_late":     "1 if previous voyage > 5 min late",
        "is_hel_to_tal":     "1 if HEL→TLL direction",
        "sched_duration_min":"Scheduled crossing duration (min)",
    }
    feat_df = pd.DataFrame([
        {"Feature": f, "Description": feat_desc.get(f, "—")}
        for f in MODEL_FEATURES
    ])
    st.dataframe(feat_df, use_container_width=True, hide_index=True)
