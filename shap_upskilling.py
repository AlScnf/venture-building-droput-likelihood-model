import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.ensemble import BalancedRandomForestClassifier

# 1. Load data
df = pd.read_csv("applicants-top-performers.csv")
y  = df["target_dropout"].astype(int)
X  = df.drop(columns=["target_dropout"])

# 2. Preprocessing
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["uint8", "int8", "bool"]).columns.tolist()
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", "passthrough", cat_cols),
])

# 3. Fit pipeline
pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", BalancedRandomForestClassifier(
        n_estimators=500,
        max_depth=4,
        max_features="sqrt",
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    ))
])
pipe.fit(X, y)

# 4. Transform entire X
X_proc    = pipe.named_steps["prep"].transform(X)
feat_names = pipe.named_steps["prep"].get_feature_names_out()

# 5. Build a SHAP Explainer on the *model* (not the pipeline)
explainer = shap.Explainer(pipe.named_steps["clf"], X_proc)

# 6. Compute SHAP values (returns an object with .values shape (n, p))
shap_out = explainer(X_proc)
shap_values = shap_out.values  # shape = (n_samples, n_features)

# 7. Find your two up-skilling feature indices
idx_cert  = list(feat_names).index("num__additional_certifications")
idx_abroad = list(feat_names).index("num__worked_abroad")

# 8. Slice to only those two columns
X_upskill = X_proc[:, [idx_cert, idx_abroad]]
shap_up    = shap_values[:, [idx_cert, idx_abroad]]

# 9. Finally, plot with labels
plt.figure(figsize=(6, 4))
shap.summary_plot(
    shap_up,
    X_upskill,
    feature_names=["Certifications", "Worked abroad"],
    show=False
)
plt.title("SHAP Impact of Up-skilling Features\n(on Dropout Probability)")
plt.xlabel("SHAP value (impact on log-odds)")
plt.tight_layout()
plt.show()
