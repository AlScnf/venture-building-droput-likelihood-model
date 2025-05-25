import pandas as pd, numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.inspection import permutation_importance

# 1. Load data
df = pd.read_csv("applicants_cleaned.csv")
y  = df["target_dropout"].astype(int)
X  = df.drop(columns=["target_dropout"])

num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["uint8","int8","bool"]).columns.tolist()

numeric_pipe = Pipeline([
    ("scale", StandardScaler()),
])

pre = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", "passthrough", cat_cols),
])

# 2. Model + tuning grid
brf = BalancedRandomForestClassifier(random_state=42, n_jobs=-1)
param_grid = {
    "clf__n_estimators":      [300, 500, 700],
    "clf__max_depth":         [3, 4],
    "clf__max_features":      ["sqrt", 0.6],
    "clf__min_samples_leaf":  [1, 3],
}

pipe = Pipeline([
    ("prep", pre),
    ("clf",  brf),
])

cv_inner = StratifiedKFold(3, shuffle=True, random_state=1)
search = GridSearchCV(
    pipe, param_grid,
    scoring="roc_auc",
    cv=cv_inner,
    refit=True,
    n_jobs=-1,
    verbose=0
)

# 3. Evaluate on full data (out-of-fold predictions)
cv_outer = StratifiedKFold(5, shuffle=True, random_state=7)
oof_proba = cross_val_predict(
    search, X, y,
    cv=cv_outer,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

print("Final CV ROC-AUC :", roc_auc_score(y, oof_proba).round(3))
print("Final CV AP      :", average_precision_score(y, oof_proba).round(3))

# 4. Train on full data to access best estimator
search.fit(X, y)
best = search.best_estimator_

# 5. Permutation importance
perm = permutation_importance(
    best, X, y,
    n_repeats=100,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1,
)


feature_names = best.named_steps["prep"].get_feature_names_out()
imp = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)
print("\n Top 15 features by permutation importance:")
print(imp.head(15).round(4))

from sklearn.metrics import precision_recall_curve, f1_score

precision, recall, thresholds = precision_recall_curve(y, oof_proba)
f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
best = np.argmax(f1)

thr      = thresholds[best]
best_f1  = f1[best]
best_p   = precision[best]
best_r   = recall[best]

print(f"\nOptimal threshold @ max F1  : {thr:.2f}")
print(f"F1 = {best_f1:.3f} | Precision = {best_p:.3f} | Recall = {best_r:.3f}")

imp.head(10).sort_values().plot.barh()

from sklearn.metrics import confusion_matrix
preds = (oof_proba >= 0.41).astype(int)
cm = confusion_matrix(y, preds)
print(cm)

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(y, (oof_proba >= 0.41).astype(int))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay", "Dropout"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix @ Threshold = 0.41")
plt.show()
