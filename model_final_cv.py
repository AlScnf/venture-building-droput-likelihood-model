import pandas as pd
import numpy as np
from collections import defaultdict

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, precision_recall_curve,
                             roc_curve, confusion_matrix)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

# LOAD DATA
df = pd.read_csv("applicants_top_performers.csv")
df = df.reset_index(drop=True)               # keep a clean row id
y  = df["target_dropout"].astype(int)
X  = df.drop(columns=["target_dropout"])

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["uint8", "int8", "bool"]).columns.tolist()

numeric_pipe = Pipeline([("scale", StandardScaler())])
pre = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", "passthrough", cat_cols)
])

# MODEL
brf  = BalancedRandomForestClassifier(random_state=42, n_jobs=-1)
pipe = Pipeline([("prep", pre), ("clf", brf)])

param_grid = {
    "clf__n_estimators":     [300, 500],
    "clf__max_depth":        [3, 4],
    "clf__max_features":     ["sqrt", 0.6],
    "clf__min_samples_leaf": [1, 3],
}

inner_cv = StratifiedKFold(3, shuffle=True, random_state=1)
outer_cv = StratifiedKFold(5, shuffle=True, random_state=7)

# Containers for predictions and feature importances
oof_proba = np.full(len(X), np.nan)

# NESTED CV
for tr_idx, te_idx in outer_cv.split(X, y):
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

    grid = GridSearchCV(pipe, param_grid, scoring="roc_auc",
                        cv=inner_cv, n_jobs=-1, refit=True)
    grid.fit(X_tr, y_tr)
    best_est = grid.best_estimator_

    calib = CalibratedClassifierCV(
        estimator=best_est,
        method="isotonic",
        cv=5
    )
    calib.fit(X_tr, y_tr)

    oof_proba[te_idx] = calib.predict_proba(X_te)[:, 1]

# METRICS
roc_auc = roc_auc_score(y, oof_proba)
ap      = average_precision_score(y, oof_proba)
brier   = brier_score_loss(y, oof_proba)

print(f"\nNested-CV ROC-AUC : {roc_auc:.3f}")
print(f"Nested-CV AP      : {ap:.3f}")
print(f"Nested-CV Brier   : {brier:.4f}")

prec, rec, thr = precision_recall_curve(y, oof_proba)
f1s   = 2 * prec * rec / (prec + rec + 1e-8)
best  = f1s.argmax()
tau   = thr[best]

print(f"\nOptimal τ (max F1) : {tau:.2f}")
print(f"F1  = {f1s[best]:.3f} | P = {prec[best]:.3f} | R = {rec[best]:.3f}")

cm = confusion_matrix(y, (oof_proba >= tau).astype(int))
print("\nConfusion matrix (rows true, cols pred):\n", cm)

# BOOTSTRAP CI
n_boot = 1000
rng     = np.random.default_rng(42)
boot = defaultdict(list)

for _ in range(n_boot):
    sample = rng.integers(0, len(y), len(y))
    if len(np.unique(y.iloc[sample])) < 2:
        continue
    boot["roc_auc"].append(roc_auc_score(y.iloc[sample], oof_proba[sample]))
    boot["ap"].append(average_precision_score(y.iloc[sample], oof_proba[sample]))
    boot["brier"].append(brier_score_loss(y.iloc[sample], oof_proba[sample]))

ci = {k: np.percentile(v, [2.5, 97.5]) for k, v in boot.items()}
print("\n95 % bootstrap CIs:")
for k, (lo, hi) in ci.items():
    print(f"{k:8s}: {lo:.3f} – {hi:.3f}")

# save bootstrap distribution
pd.DataFrame(boot).to_csv("bootstrap_metrics.csv", index=False)

# ─────────────────────────────────────────────────────────────  6. PLOTS
plt.figure(figsize=(13, 10))

# 6a ROC curve
plt.subplot(2, 2, 1)
fpr, tpr, _ = roc_curve(y, oof_proba)
plt.plot(fpr, tpr, label=f"AUC {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC curve"); plt.legend()

# 6b PR curve
plt.subplot(2, 2, 2)
plt.plot(rec, prec, label=f"AP {ap:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision–Recall curve"); plt.legend()

# 6c Calibration curve (10 quantile bins)
plt.subplot(2, 2, 3)
prob_true, prob_pred = calibration_curve(y, oof_proba, n_bins=10, strategy="quantile")
plt.plot(prob_pred, prob_true, marker="o")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("Mean predicted"); plt.ylabel("Fraction positives")
plt.title("Calibration (quantile bins)")

# 6d Confusion matrix heat-map
plt.subplot(2, 2, 4)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Stay", "Drop"], yticklabels=["Stay", "Drop"])
plt.title(f"Confusion matrix @ τ={tau:.2f}")

plt.tight_layout()
plt.savefig("model_plots.png", dpi=300)
plt.show()

# ─────────────  7. CUMULATIVE-GAIN / LIFT
df_preds = pd.DataFrame({
    "id": df.index,
    "truth": y,
    "proba": oof_proba
}).sort_values("proba", ascending=False).reset_index(drop=True)

df_preds["cum_churn"] = df_preds["truth"].cumsum()
total_churn = df_preds["truth"].sum()
df_preds["gain"] = df_preds["cum_churn"] / total_churn

plt.figure(figsize=(6, 4))
plt.plot(np.arange(1, len(df_preds) + 1) / len(df_preds),
         df_preds["gain"], label="Model")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Random")
plt.xlabel("Proportion reviewed"); plt.ylabel("Proportion of churn captured")
plt.title("Cumulative-gain curve"); plt.legend()
plt.tight_layout(); plt.savefig("gain_curve.png", dpi=300); plt.show()

# quick top-k table for thesis
for k in [10, 20, 30, int(len(df)*0.3)]:
    top = df_preds.head(k)
    prec_k = top["truth"].mean()
    recall_k = top["truth"].sum() / total_churn
    print(f"Top {k:>3}  |  precision {prec_k:.3f}  |  recall {recall_k:.3f}")

# ─────────────  8. EXPORT OOF PROBABILITIES
df["dropout_proba"] = oof_proba
df.to_csv("oof_predictions.csv", index=False)

print("\nSaved:")
print(" • model_plots.png")
print(" • gain_curve.png")
print(" • oof_predictions.csv")
print(" • bootstrap_metrics.csv")

from sklearn.metrics import brier_score_loss

# Before calibration
raw_proba = best.predict_proba(X_test)[:, 1]
brier_raw = brier_score_loss(y_test, raw_proba)

# After calibration
calib = CalibratedClassifierCV(best, method="isotonic", cv=5)
calib.fit(X_train, y_train)
calib_proba = calib.predict_proba(X_test)[:, 1]
brier_calib = brier_score_loss(y_test, calib_proba)

print(brier_raw, brier_calib)