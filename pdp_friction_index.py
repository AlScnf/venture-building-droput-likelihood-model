import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split

# 1. Load your data
df = pd.read_csv("applicants_top_performers.csv")

# 2. Prepare features and target
y = df["target_dropout"].astype(int)
X = df.drop(columns=["target_dropout"])

# 3. Identify numeric and categorical columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["uint8", "int8", "bool"]).columns.tolist()

# 4. Build preprocessing + model pipeline
numeric_pipe = Pipeline([("scale", StandardScaler())])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", "passthrough", cat_cols),
])

model = Pipeline([
    ("prep", preprocessor),
    ("rf", RandomForestClassifier(random_state=42, n_jobs=-1))
])

# 5. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42
)

model.fit(X_train, y_train)

# 6. Identify top 4 features for partial dependence
importances = model.named_steps["rf"].feature_importances_
feature_names = model.named_steps["prep"].get_feature_names_out()
top_indices = np.argsort(importances)[-4:]

# 7. Plot Partial Dependence Curves
fig, ax = plt.subplots(figsize=(12, 8))
PartialDependenceDisplay.from_estimator(
    model,
    X_test,
    features=top_indices,
    feature_names=feature_names,
    ax=ax
)
plt.tight_layout()
plt.show()
