# vif_plot.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ── 1. Load the cleaned dataset
df = pd.read_csv("cleaned.csv")

# ── 2. Keep only numeric predictors
X_num = df.select_dtypes(include=["int64", "float64"]).copy()

# Add intercept for VIF calc
X_num["intercept"] = 1

# ── 3. Compute VIF for each numeric column
vif = pd.DataFrame()
vif["Feature"] = X_num.columns
vif["VIF"] = [variance_inflation_factor(X_num.values, i)
              for i in range(X_num.shape[1])]

# Drop the intercept row
vif = vif[vif["Feature"] != "intercept"]

# ── 4. Stats you may cite
vif_max   = vif["VIF"].max()
vif_median = vif["VIF"].median()
print("Max VIF :", round(vif_max, 2))
print("Median VIF :", round(vif_median, 2))

# ── 5. Plot histogram
sns.set(style="whitegrid")
plt.figure(figsize=(7,4))
sns.histplot(vif["VIF"], bins=15, color="#f0b949", kde=True)
plt.axvline(vif_max,   color="red",   ls="--", label=f"Max VIF = {vif_max:.2f}")
plt.axvline(vif_median,color="green", ls="--", label=f"Median VIF = {vif_median:.2f}")
plt.xlabel("VIF Value")
plt.ylabel("Frequency")
plt.title("Distribution of VIF across numeric features")
plt.legend()
plt.tight_layout()
plt.savefig("vif_histogram.png", dpi=300)
plt.show()
