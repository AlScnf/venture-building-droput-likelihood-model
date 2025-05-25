import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay

# 1. Load data
df = pd.read_csv("applicants_top_performers.csv")
y  = df["target_dropout"].astype(int)
X  = df.drop(columns=["target_dropout"])

# 2. Define preprocessing pipeline
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["uint8", "int8", "bool"]).columns.tolist()

numeric_pipe = Pipeline([("scale", StandardScaler())])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", "passthrough", cat_cols),
])

# 3. Build and fit the model pipeline
pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", BalancedRandomForestClassifier(random_state=42, n_jobs=-1))
])
pipe.fit(X, y)

# 4. Get the transformed feature names
feature_names = pipe.named_steps["prep"].get_feature_names_out()

# 5. Locate indices of the desired features
desired = ["promotions_rate", "current_tenure", "month_of_experience"]
feature_indices = [
    list(feature_names).index(f"num__{name}") for name in desired
]

# 6. Plot partial dependence for the three features
fig, ax = plt.subplots(figsize=(12, 6))
PartialDependenceDisplay.from_estimator(
    pipe,
    X,
    features=feature_indices,
    feature_names=feature_names,
    ax=ax
)
ax.set_title("Partial Dependence of Dropout on Career Momentum Features")
plt.tight_layout()
plt.show()
