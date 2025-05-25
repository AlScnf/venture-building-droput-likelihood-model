import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import (
    StratifiedKFold, train_test_split, RandomizedSearchCV, cross_validate
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, confusion_matrix
)
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBClassifier
from scipy.stats import uniform, randint

# ------- 1. Load -------------------------------------------------------------
FILE = Path("applicants_cleaned.csv")
df   = pd.read_csv(FILE)

TARGET = "target_dropout"              # 1 = dropped, 0 = stayed
y = df[TARGET].astype(int)
X = df.drop(columns=[TARGET])

# ------- 2. Column splits ----------------------------------------------------
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["uint8", "int8", "bool", "object"]).columns.tolist()

# sanity: object columns should be dummy-free. If any, one-hot them here.
if any(X[c].dtype == "object" for c in cat_cols):
    raise ValueError("Object dtype found – convert to dummies first!")

# ------- 3. Pre-processing pipeline -----------------------------------------
numeric_pipe = Pipeline([
    ("varth", VarianceThreshold(threshold=0.0)),   # drop constant cols
    ("sc", StandardScaler()),
])

pre = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", "passthrough", cat_cols),
])

# ------- 4. Baseline – penalised logistic -----------------------------------
log_reg = Pipeline([
    ("prep", pre),
    ("clf",  LogisticRegression(
                max_iter=400,
                solver="liblinear",
                class_weight="balanced",
                penalty="l2",
                C=1.0,
            )),
])

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
baseline_scores = cross_validate(
    log_reg, X, y,
    cv=outer_cv,
    scoring=["roc_auc", "average_precision", "f1", "accuracy"],
    n_jobs=-1,
)
print("\n=== Logistic baseline (CV mean ± sd) ===")
for k, v in baseline_scores.items():
    if "test" in k:
        print(f"{k:20s}: {v.mean():.3f} ± {v.std():.3f}")

# ------- 5. Main model – XGBoost with nested CV + random search -------------
xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
)

search_space = {
    "clf__n_estimators":    randint(50, 250),
    "clf__max_depth":       randint(2, 6),
    "clf__learning_rate":   uniform(0.02, 0.18),
    "clf__subsample":       uniform(0.6, 0.4),
    "clf__colsample_bytree":uniform(0.6, 0.4),
    "clf__min_child_weight":randint(1, 8),
    "clf__gamma":           uniform(0, 2),
}

xgb_pipe = Pipeline([
    ("prep", pre),
    ("clf",  xgb),
])

inner_cv  = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
rand_search = RandomizedSearchCV(
    estimator=xgb_pipe,
    param_distributions=search_space,
    n_iter=60,
    scoring="roc_auc",
    cv=inner_cv,
    n_jobs=-1,
    refit=True,
    verbose=0,
    random_state=1,
)

nested_scores = cross_validate(
    rand_search, X, y,
    cv=outer_cv,
    scoring=["roc_auc", "average_precision", "f1", "accuracy"],
    n_jobs=-1,
    return_estimator=True,
)

print("\n=== XGBoost nested-CV results (mean ± sd) ===")
for k, v in nested_scores.items():
    if "test" in k:
        print(f"{k:20s}: {v.mean():.3f} ± {v.std():.3f}")

