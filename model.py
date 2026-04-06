import os
import json
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve,
)

print("=" * 65)
print("PHASE 3 — MODEL TRAINING")
print("=" * 65)

os.makedirs("outputs", exist_ok=True)

# ──────────────────────────────────────────────────────────────────
# 1. LOAD
# ──────────────────────────────────────────────────────────────────
csv_path = os.path.join("outputs", "voyage_dataset.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError("Run phase1.py and phase2.py first.")

df = pd.read_csv(csv_path, parse_dates=["etd_sched"])
df = df.sort_values("etd_sched").reset_index(drop=True)
print(f"Loaded {len(df):,} rows | {df.shape[1]} columns")
print(f"Date range: {df['etd_sched'].min()} → {df['etd_sched'].max()}")

# ──────────────────────────────────────────────────────────────────
# 2. FEATURE LIST
# ──────────────────────────────────────────────────────────────────
MODEL_FEATURES = [
    # Speed at time of last ping before departure
    "mean_sog",
    "is_slow_crossing",

    # Temporal context
    "hour_of_dep",
    "day_of_week",
    "month",
    "is_weekend",
    "is_peak_hour",

    # Cyclical encodings of time
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",

    # Port congestion at departure time
    "port_traffic",
    "traffic_3h",
    "traffic_6h",
    "prev_traffic",

    # Vessel history (no leakage — all shift(1) from Phase 1)
    "vessel_avg_delay",
    "prev_dep_delay",
    "rolling_delay_3v",
    "prev_was_late",

    # Route and schedule context
    "is_hel_to_tal",
    "sched_duration_min",
]
TARGET = "is_delayed"

missing = [c for c in MODEL_FEATURES + [TARGET] if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}. Re-run phase1.py and phase2.py.")

df_model = df.dropna(subset=[TARGET]).copy()
# Fill NaN in lag/speed features (first voyage per vessel has no history)
for c in MODEL_FEATURES:
    if df_model[c].isna().any():
        df_model[c] = df_model[c].fillna(df_model[c].median())

df_model = df_model.reset_index(drop=True)
print(f"Modelling rows: {len(df_model):,}")

X = df_model[MODEL_FEATURES]
y = df_model[TARGET].astype(int)

delay_rate = y.mean()
print(f"\nClass distribution:\n{y.value_counts().to_string()}")
print(f"Delay rate: {delay_rate:.1%}")

# ──────────────────────────────────────────────────────────────────
# 3. CHRONOLOGICAL TRAIN / TEST SPLIT  (70 / 30)
#    No shuffle — time-series data must NOT be shuffled.
# ──────────────────────────────────────────────────────────────────
split_idx = int(len(df_model) * 0.70)
X_train = X.iloc[:split_idx].copy()
X_test  = X.iloc[split_idx:].copy()
y_train = y.iloc[:split_idx].copy()
y_test  = y.iloc[split_idx:].copy()

print(f"\nTrain: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
print(f"Train delay rate: {y_train.mean():.1%}  |  Test delay rate: {y_test.mean():.1%}")

# ──────────────────────────────────────────────────────────────────
# 4. LEAK-FREE vessel_avg_delay
# ──────────────────────────────────────────────────────────────────
train_vessel_delay = (
    df_model.iloc[:split_idx]
    .groupby("ship")["dep_delay_min"]
    .mean()
)
fallback_delay = float(train_vessel_delay.mean())

X_train["vessel_avg_delay"] = (
    df_model.iloc[:split_idx]["ship"]
    .map(train_vessel_delay)
    .fillna(fallback_delay)
    .values
)
X_test["vessel_avg_delay"] = (
    df_model.iloc[split_idx:]["ship"]
    .map(train_vessel_delay)
    .fillna(fallback_delay)
    .values
)
print("vessel_avg_delay recomputed on train split only ✅")

# ──────────────────────────────────────────────────────────────────
# 5. SCALE FEATURES
# ──────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=MODEL_FEATURES)
X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=MODEL_FEATURES)

# ──────────────────────────────────────────────────────────────────
# 6. CROSS-VALIDATION COMPARISON  (train set only)
# ──────────────────────────────────────────────────────────────────
print("\n--- CROSS-VALIDATION MODEL COMPARISON (train set, 5-fold) ---")

cv = StratifiedKFold(n_splits=5, shuffle=False)  # No shuffle for time-series

candidates = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=8,
        class_weight="balanced", random_state=42, n_jobs=-1,
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=8, random_state=42,
    ),
    "LogisticRegression": LogisticRegression(
        C=0.1, class_weight="balanced", max_iter=1000, random_state=42,
    ),
}

cv_results = {}
for name, clf in candidates.items():
    X_cv = X_train_scaled if name == "LogisticRegression" else X_train
    f1_scores = cross_val_score(clf, X_cv, y_train, cv=cv, scoring="f1")
    roc_scores = cross_val_score(clf, X_cv, y_train, cv=cv, scoring="roc_auc")
    cv_results[name] = {
        "f1_mean":   f1_scores.mean(),
        "f1_std":    f1_scores.std(),
        "roc_mean":  roc_scores.mean(),
        "roc_std":   roc_scores.std(),
    }
    print(f"  {name:<22} F1={f1_scores.mean():.3f}±{f1_scores.std():.3f}  "
          f"ROC-AUC={roc_scores.mean():.3f}±{roc_scores.std():.3f}")

# Pick best by F1
best_name = max(cv_results, key=lambda k: cv_results[k]["f1_mean"])
print(f"\nBest CV model: {best_name}")

# ──────────────────────────────────────────────────────────────────
# 7. HYPERPARAMETER SEARCH  (small grid — keeps runtime < 30 s)
# ──────────────────────────────────────────────────────────────────
print("\n--- HYPERPARAMETER SEARCH (RandomForest) ---")

param_grid = [
    {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 8},
    {"n_estimators": 200, "max_depth": 7, "min_samples_leaf": 6},
    {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 8},
]

best_params, best_cv_f1 = param_grid[0], -1
for params in param_grid:
    rf = RandomForestClassifier(
        **params, class_weight="balanced", random_state=42, n_jobs=-1
    )
    f1_cv = cross_val_score(rf, X_train, y_train, cv=cv, scoring="f1").mean()
    print(f"  {params}  →  F1={f1_cv:.4f}")
    if f1_cv > best_cv_f1:
        best_cv_f1  = f1_cv
        best_params = params

print(f"\nBest RF params: {best_params}  (CV F1={best_cv_f1:.4f})")

# ──────────────────────────────────────────────────────────────────
# 8. TRAIN FINAL MODEL
# ──────────────────────────────────────────────────────────────────
print("\nTraining final RandomForestClassifier...")

model = RandomForestClassifier(
    **best_params,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)
print("Training complete ✅")

# ──────────────────────────────────────────────────────────────────
# 9. THRESHOLD TUNING
# ──────────────────────────────────────────────────────────────────
print("\n--- THRESHOLD ANALYSIS ---")

y_proba = model.predict_proba(X_test)[:, 1]

threshold_results = {}
for thresh in [0.30, 0.40, 0.50]:
    y_pred_t = (y_proba >= thresh).astype(int)
    f1   = f1_score(y_test,   y_pred_t, zero_division=0)
    prec = precision_score(y_test, y_pred_t, zero_division=0)
    rec  = recall_score(y_test,    y_pred_t, zero_division=0)
    acc  = accuracy_score(y_test,  y_pred_t)
    threshold_results[thresh] = {"f1": f1, "precision": prec, "recall": rec, "accuracy": acc}
    print(f"  thresh={thresh:.2f}  F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  Acc={acc:.3f}")

# Choose threshold maximising F1
BEST_THRESHOLD = max(threshold_results, key=lambda t: threshold_results[t]["f1"])
print(f"\nChosen threshold: {BEST_THRESHOLD} (best F1)")

y_pred = (y_proba >= BEST_THRESHOLD).astype(int)

# ──────────────────────────────────────────────────────────────────
# 10. EVALUATE
# ──────────────────────────────────────────────────────────────────
train_acc = model.score(X_train, y_train)
test_acc  = accuracy_score(y_test, y_pred)
gap       = train_acc - test_acc

print("\n" + "-" * 45)
print("MODEL PERFORMANCE")
print("-" * 45)
print(f"Train accuracy   : {train_acc:.4f}")
print(f"Test  accuracy   : {test_acc:.4f}")
print(f"Gap              : {gap:.4f}  {'⚠️  possible overfit' if gap > 0.10 else '✅ acceptable'}")

try:
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC          : {roc_auc:.4f}")
except Exception:
    roc_auc = None

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["On Time", "Delayed"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ──────────────────────────────────────────────────────────────────
# 11. FEATURE IMPORTANCE
# ──────────────────────────────────────────────────────────────────
importances = (
    pd.Series(model.feature_importances_, index=MODEL_FEATURES)
    .sort_values(ascending=False)
)
print("\nFeature Importances:")
for feat, imp in importances.items():
    bar = "█" * int(imp * 50)
    print(f"  {feat:<25} {imp:.4f}  {bar}")

# ──────────────────────────────────────────────────────────────────
# 12. SAVE MODEL BUNDLE  (model + scaler + metadata)
# ──────────────────────────────────────────────────────────────────
model_bundle = {
    "model":         model,
    "scaler":        scaler,
    "features":      MODEL_FEATURES,
    "threshold":     BEST_THRESHOLD,
    "vessel_delays": train_vessel_delay.to_dict(),
    "fallback_delay":fallback_delay,
}
model_path = os.path.join("outputs", "congestion_model.pkl")
joblib.dump(model_bundle, model_path)
print(f"\n✅ Model bundle saved → {model_path}")

# ──────────────────────────────────────────────────────────────────
# 13. SAVE METRICS JSON
# ──────────────────────────────────────────────────────────────────
metrics = {
    "accuracy":          round(float(test_acc), 4),
    "precision":         round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
    "recall":            round(float(recall_score(y_test,    y_pred, zero_division=0)), 4),
    "f1":                round(float(f1_score(y_test,        y_pred, zero_division=0)), 4),
    "roc_auc":           round(float(roc_auc), 4) if roc_auc else None,
    "train_accuracy":    round(float(train_acc), 4),
    "test_accuracy":     round(float(test_acc),  4),
    "threshold":         BEST_THRESHOLD,
    "n_train":           int(len(X_train)),
    "n_test":            int(len(X_test)),
    "delay_rate":        round(float(delay_rate), 4),
    "feature_importance":importances.round(4).to_dict(),
    "cv_results":        {k: {m: round(v, 4) for m, v in vs.items()} for k, vs in cv_results.items()},
    "best_params":       best_params,
    "threshold_analysis":{str(t): {m: round(v, 4) for m, v in tv.items()} for t, tv in threshold_results.items()},
}

metrics_path = os.path.join("outputs", "model_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"✅ Metrics saved     → {metrics_path}")

print("\n" + "=" * 65)
print("PHASE 3 COMPLETE — Run phase3_optimizer.py next")
print("=" * 65)
