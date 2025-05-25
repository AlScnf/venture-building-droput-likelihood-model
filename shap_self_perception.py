import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.ensemble import BalancedRandomForestClassifier

# 1. Load & prepare
df = pd.read_csv("applicants_top_performers.csv")
y  = df["target_dropout"].astype(int)
X  = df.drop(columns=["target_dropout"])

num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["uint8","int8","bool"]).columns.tolist()
pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat","passthrough",   cat_cols),
])

pipe = Pipeline([
    ("prep", pre),
    ("clf", BalancedRandomForestClassifier(
        n_estimators=500, max_depth=4,
        max_features="sqrt", min_samples_leaf=1,
        random_state=42, n_jobs=-1
    ))
])
pipe.fit(X, y)

# 2. Transform & get names
X_proc    = pipe.named_steps["prep"].transform(X)
feat_names = pipe.named_steps["prep"].get_feature_names_out()

# 3. Build SHAP Explainer
explainer = shap.Explainer(pipe.named_steps["clf"], X_proc)
shap_out   = explainer(X_proc)

# 4. Main-effect beeswarm for the two self-perception features
idx_grad  = list(feat_names).index("num__self_def_recent_graduate")
idx_found = list(feat_names).index("num__how_many_time_founder")

plt.figure(figsize=(6, 4))
shap.summary_plot(
    shap_out.values[:, [idx_grad, idx_found]],
    X_proc[:, [idx_grad, idx_found]],
    feature_names=["Recent Graduate","Founder Attempts"],
    show=False
)
plt.title("SHAP Main Effects: Self-Perception Features")
plt.tight_layout()
plt.savefig("A4_main_effects.png", dpi=200)
plt.show()

# 5. Interaction plot between those two features
#    Use TreeExplainer directly to get interaction values
te = shap.TreeExplainer(pipe.named_steps["clf"])
shap_ints = te.shap_interaction_values(X_proc)
# If binary, take class-1 interactions
shap_int = shap_ints[1] if isinstance(shap_ints, list) else shap_ints

plt.figure(figsize=(6, 4))
shap.dependence_plot(
    (idx_grad, idx_found),
    shap_int,
    X_proc,
    feature_names=feat_names,
    show=False
)
plt.title("SHAP Interaction: Recent Graduate × Founder Attempts")
plt.tight_layout()
plt.savefig("A4_interaction.png", dpi=200)
plt.show()
