import pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import (StratifiedKFold,
                                     GridSearchCV, RandomizedSearchCV,
                                     cross_validate)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import make_scorer, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier   # pip install imbalanced-learn
from catboost import CatBoostClassifier                      # pip install catboost
from scipy.stats import uniform, randint

# ------------------ 0. Load -------------------------------------------------
df = pd.read_csv("applicants-cleaned.csv")
y  = df["target_dropout"].astype(int)
X  = df.drop(columns=["target_dropout"])

num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["uint8","int8","bool"]).columns.tolist()

numeric_pipe = Pipeline([
    ("varth", VarianceThreshold()),   # drop constants
    ("sc",    StandardScaler()),
])

pre = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", "passthrough", cat_cols),
])

outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ------------------ 1. Model zoo -------------------------------------------
models = {}

# 1. ElasticNet logistic
log_clf = LogisticRegression(max_iter=500, solver="saga",
                             class_weight="balanced", n_jobs=-1)
param_log = {
    "clf__penalty": ["l1", "l2", "elasticnet"],
    "clf__C": np.logspace(-3, 1, 9),
    "clf__l1_ratio": [0.1, 0.5, 0.9]
}
models["Logistic"] = ("grid", log_clf, param_log)

# 2. HistGradientBoosting
hgb = HistGradientBoostingClassifier(
        max_depth=None, learning_rate=0.05, l2_regularization=0.1,
        max_iter=250, class_weight="balanced", random_state=42)
param_hgb = {
    "clf__max_depth": [None, 2, 3],
    "clf__l2_regularization": [0.05, 0.1, 0.2],
    "clf__max_leaf_nodes": [15, 31],
    "clf__learning_rate": [0.03, 0.05, 0.08],
}
models["HistGB"] = ("grid", hgb, param_hgb)

# 3. CatBoost (no scaling needed → but we still pass through)
cat = CatBoostClassifier(
        iterations=200, depth=4, learning_rate=0.05,
        eval_metric="AUC", verbose=False, class_weights=[1,1],
        random_state=42, loss_function="Logloss")
param_cat = {
    "clf__depth": [3,4],
    "clf__l2_leaf_reg": [1,3,5],
    "clf__learning_rate": [0.03,0.05,0.08],
}
models["CatBoost"] = ("grid", cat, param_cat)

# 4. Balanced Random Forest
brf = BalancedRandomForestClassifier(
        n_estimators=400, max_depth=4, random_state=42)
param_brf = {
    "clf__n_estimators": [300,400,600],
    "clf__max_depth": [3,4],
    "clf__max_features": ["sqrt", 0.6, 0.8],
}
models["BalancedRF"] = ("grid", brf, param_brf)

# ------------------ 2. Sweep loop ------------------------------------------
leaderboard = []

for name, (search_type, base_clf, grid) in models.items():
    pipe = Pipeline([("prep", pre), ("clf", base_clf)])
    if search_type == "grid":
        search = GridSearchCV(pipe, grid, scoring="roc_auc",
                              cv=StratifiedKFold(3, shuffle=True, random_state=1),
                              n_jobs=-1, refit=True, verbose=0)
    else:   # random: (estimator, grid)
        search = RandomizedSearchCV(pipe, grid, n_iter=40, scoring="roc_auc",
                                    cv=StratifiedKFold(3, shuffle=True, random_state=1),
                                    n_jobs=-1, refit=True, verbose=0, random_state=1)
    cv_res = cross_validate(search, X, y, cv=outer,
                            scoring=["roc_auc", "average_precision"],
                            return_estimator=False, n_jobs=-1)
    leaderboard.append({
        "model": name,
        "roc_auc_mean": cv_res["test_roc_auc"].mean(),
        "roc_auc_sd":   cv_res["test_roc_auc"].std(),
        "ap_mean":      cv_res["test_average_precision"].mean(),
        "ap_sd":        cv_res["test_average_precision"].std(),
    })

lb = (pd.DataFrame(leaderboard)
        .sort_values("roc_auc_mean", ascending=False)
        .reset_index(drop=True))
print("\n=== Leaderboard (nested 5-fold) ===")
print(lb.to_string(index=False, float_format="%.3f"))
